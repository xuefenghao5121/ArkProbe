"""eBPF-based system-level observation using BCC and bpftrace.

Provides non-intrusive system-level tracing for:
- Block I/O latency histograms
- Lock contention analysis
- Off-CPU analysis
- TCP latency
- Cache statistics
- Memory access patterns
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseCollector, CollectionResult
from ..utils.process import RunResult, run_cmd, run_shell

log = logging.getLogger(__name__)


@dataclass
class HistogramBucket:
    """A single histogram bucket."""
    low: int
    high: int
    count: int


@dataclass
class LatencyHistogram:
    """Latency distribution as histogram buckets."""
    unit: str  # "us", "ms", "ns"
    buckets: List[HistogramBucket] = field(default_factory=list)

    @property
    def avg(self) -> float:
        total_count = sum(b.count for b in self.buckets)
        if total_count == 0:
            return 0.0
        weighted = sum(((b.low + b.high) / 2) * b.count for b in self.buckets)
        return weighted / total_count

    @property
    def p99(self) -> float:
        total_count = sum(b.count for b in self.buckets)
        if total_count == 0:
            return 0.0
        target = total_count * 0.99
        cumulative = 0
        for b in self.buckets:
            cumulative += b.count
            if cumulative >= target:
                return float(b.high)
        return float(self.buckets[-1].high) if self.buckets else 0.0


class EbpfCollector(BaseCollector):
    """eBPF-based system-level observation using BCC tools and bpftrace."""

    def __init__(self, output_dir: Path, backend: str = "auto"):
        super().__init__(output_dir)
        self.backend = backend
        self._bpftrace_available: Optional[bool] = None
        self._bcc_available: Optional[bool] = None

    def collect(self, pid: Optional[int] = None,
                duration_sec: int = 30,
                probes: Optional[List[str]] = None,
                **kwargs) -> CollectionResult:
        """Run all configured eBPF probes."""
        if probes is None:
            probes = ["io_latency", "lock_contention", "offcpu",
                      "cache_stats", "tcp_latency"]

        data: Dict[str, Any] = {}
        errors = []
        raw_files = {}

        probe_methods = {
            "io_latency": self.trace_io_latency,
            "lock_contention": self.trace_lock_contention,
            "offcpu": self.trace_offcpu,
            "cache_stats": self.trace_cache_stats,
            "tcp_latency": self.trace_tcp_latency,
            "network_stats": self.trace_network_stats,
        }

        for probe_name in probes:
            method = probe_methods.get(probe_name)
            if method is None:
                errors.append(f"Unknown probe: {probe_name}")
                continue
            try:
                self.log.info("Running eBPF probe: %s", probe_name)
                result = method(duration_sec=duration_sec, pid=pid)
                data[probe_name] = result
                raw_path = self._save_raw(
                    f"ebpf_{probe_name}.json",
                    json.dumps(result, indent=2, default=str),
                )
                raw_files[probe_name] = raw_path
            except Exception as e:
                self.log.error("eBPF probe %s failed: %s", probe_name, e)
                errors.append(f"{probe_name}: {e}")

        return CollectionResult(
            collector_name="ebpf",
            data=data,
            raw_files=raw_files,
            errors=errors,
        )

    def trace_io_latency(
        self, duration_sec: int = 30, pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Trace block I/O latency distribution using biolatency (BCC)."""
        cmd = ["biolatency-bpfcc", "-j", str(duration_sec), "1"]
        if pid is not None:
            cmd = ["biolatency-bpfcc", "-j", "-p", str(pid), str(duration_sec), "1"]

        result = run_cmd(cmd, timeout_sec=duration_sec + 30)
        histogram = self._parse_bcc_histogram(result.stdout)

        return {
            "histogram": [{"low": b.low, "high": b.high, "count": b.count}
                          for b in histogram.buckets],
            "avg_latency_us": histogram.avg,
            "p99_latency_us": histogram.p99,
            "unit": "us",
        }

    def trace_lock_contention(
        self, duration_sec: int = 30, pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Trace mutex/futex lock contention."""
        # Use bpftrace for lock contention
        program = r"""
        kprobe:mutex_lock { @start[tid] = nsecs; }
        kretprobe:mutex_lock /@start[tid]/ {
            @lock_ns = hist(nsecs - @start[tid]);
            @total_wait += nsecs - @start[tid];
            @lock_count++;
            delete(@start[tid]);
        }
        interval:s:1 { @elapsed++; }
        """
        result = self._run_bpftrace(program, duration_sec)

        total_wait_ns = self._extract_bpftrace_var(result, "total_wait")
        lock_count = self._extract_bpftrace_var(result, "lock_count")
        elapsed = max(self._extract_bpftrace_var(result, "elapsed"), 1)

        return {
            "total_wait_ns": total_wait_ns,
            "lock_count": lock_count,
            "avg_wait_ns": total_wait_ns / max(lock_count, 1),
            "lock_contention_pct": (total_wait_ns / (elapsed * 1e9)) * 100,
        }

    def trace_offcpu(
        self, duration_sec: int = 30, pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Trace off-CPU time to identify where threads wait."""
        cmd = ["offcputime-bpfcc", "-f", str(duration_sec)]
        if pid is not None:
            cmd = ["offcputime-bpfcc", "-f", "-p", str(pid), str(duration_sec)]

        result = run_cmd(cmd, timeout_sec=duration_sec + 30)

        # Parse folded stacks output
        stacks: Dict[str, int] = {}
        total_offcpu_us = 0
        for line in result.stdout.splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    us = int(parts[1])
                    stacks[parts[0]] = us
                    total_offcpu_us += us
                except ValueError:
                    continue

        # Top off-CPU stacks
        top_stacks = sorted(stacks.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_offcpu_us": total_offcpu_us,
            "top_stacks": [{"stack": s, "time_us": t} for s, t in top_stacks],
        }

    def trace_cache_stats(
        self, duration_sec: int = 30, pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Collect page cache hit/miss statistics using cachestat."""
        cmd = ["cachestat-bpfcc", str(duration_sec), "1"]
        result = run_cmd(cmd, timeout_sec=duration_sec + 30)

        total_hits = 0
        total_misses = 0
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    total_hits += int(parts[0])
                    total_misses += int(parts[1])
                except ValueError:
                    continue

        total = total_hits + total_misses
        return {
            "page_cache_hits": total_hits,
            "page_cache_misses": total_misses,
            "hit_rate": total_hits / total if total > 0 else 0.0,
        }

    def trace_tcp_latency(
        self, duration_sec: int = 30, pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Trace TCP connection and round-trip latencies."""
        cmd = ["tcprtt-bpfcc", "-d", str(duration_sec)]
        if pid is not None:
            cmd.extend(["-p", str(pid)])

        result = run_cmd(cmd, timeout_sec=duration_sec + 30)

        # Fallback to tcpconnlat if tcprtt not available
        if not result.ok:
            cmd = ["tcpconnlat-bpfcc", str(duration_sec)]
            if pid:
                cmd = ["tcpconnlat-bpfcc", "-p", str(pid), str(duration_sec)]
            result = run_cmd(cmd, timeout_sec=duration_sec + 30)

        latencies = []
        for line in result.stdout.splitlines():
            parts = line.split()
            # Look for latency value (usually last column, in ms)
            if parts and re.match(r"[\d.]+", parts[-1]):
                try:
                    latencies.append(float(parts[-1]) * 1000)  # ms -> us
                except ValueError:
                    continue

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        latencies.sort()
        p99_lat = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

        return {
            "avg_latency_us": avg_lat,
            "p99_latency_us": p99_lat,
            "sample_count": len(latencies),
        }

    def trace_network_stats(
        self, duration_sec: int = 30, pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Collect network packet rates and sizes."""
        # Read /proc/net/dev before and after
        def read_net_dev() -> Dict[str, Dict[str, int]]:
            stats = {}
            try:
                content = Path("/proc/net/dev").read_text()
                for line in content.splitlines()[2:]:
                    parts = line.split()
                    if len(parts) >= 17:
                        iface = parts[0].rstrip(":")
                        stats[iface] = {
                            "rx_bytes": int(parts[1]),
                            "rx_packets": int(parts[2]),
                            "tx_bytes": int(parts[9]),
                            "tx_packets": int(parts[10]),
                        }
            except Exception:
                pass
            return stats

        before = read_net_dev()
        import time
        time.sleep(duration_sec)
        after = read_net_dev()

        # Aggregate across all non-loopback interfaces
        total = {"rx_bytes": 0, "rx_packets": 0, "tx_bytes": 0, "tx_packets": 0}
        for iface in after:
            if iface == "lo" or iface not in before:
                continue
            for key in total:
                total[key] += after[iface][key] - before[iface].get(key, 0)

        return {
            "packets_per_sec_rx": total["rx_packets"] / duration_sec,
            "packets_per_sec_tx": total["tx_packets"] / duration_sec,
            "bandwidth_rx_mbps": (total["rx_bytes"] * 8) / (duration_sec * 1e6),
            "bandwidth_tx_mbps": (total["tx_bytes"] * 8) / (duration_sec * 1e6),
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _run_bpftrace(self, program: str, duration_sec: int) -> str:
        """Write a bpftrace program to a temp file and execute it."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bt", delete=False) as f:
            f.write(program)
            f.flush()
            result = run_cmd(
                ["bpftrace", f.name],
                timeout_sec=duration_sec + 10,
            )
            return result.stdout + result.stderr

    def _extract_bpftrace_var(self, output: str, var_name: str) -> int:
        """Extract a scalar variable value from bpftrace output."""
        m = re.search(rf"@{var_name}:\s*(\d+)", output)
        return int(m.group(1)) if m else 0

    def _parse_bcc_histogram(self, output: str) -> LatencyHistogram:
        """Parse BCC tool histogram output (log2 buckets)."""
        buckets = []
        for line in output.splitlines():
            # Match lines like: [4, 8)   123 |@@@@@@|
            m = re.match(r"\s*\[(\d+),\s*(\d+)\)\s+(\d+)", line)
            if m:
                buckets.append(HistogramBucket(
                    low=int(m.group(1)),
                    high=int(m.group(2)),
                    count=int(m.group(3)),
                ))
        return LatencyHistogram(unit="us", buckets=buckets)

    def _check_bpftrace(self) -> bool:
        """Check if bpftrace is available."""
        if self._bpftrace_available is None:
            result = run_cmd(["which", "bpftrace"], timeout_sec=5)
            self._bpftrace_available = result.ok
        return self._bpftrace_available

    def _check_bcc(self) -> bool:
        """Check if BCC tools are available."""
        if self._bcc_available is None:
            result = run_cmd(["which", "biolatency-bpfcc"], timeout_sec=5)
            self._bcc_available = result.ok
        return self._bcc_available
