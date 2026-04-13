"""Linux perf wrapper for ARM PMU event collection on Kunpeng processors.

Supports:
- perf stat with CSV output (-x ,) for broad version compatibility
- perf record for sampling-based profiling
- ARM TopDown Level 1 analysis
- Sequential event group collection to avoid multiplexing
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .arm_events import (
    CORE_EVENT_GROUPS,
    UNCORE_EVENT_GROUPS,
    EventGroup,
    build_perf_event_string,
    get_all_core_event_groups,
    get_kunpeng_model,
    resolve_uncore_events,
)
from .base import BaseCollector, CollectionResult
from ..utils.process import RunResult, run_cmd

log = logging.getLogger(__name__)

# Dangerous shell metacharacters that could enable command injection
DANGEROUS_CHARS = set(";&|`$()<>\\\\")


def validate_command_safety(command: str) -> tuple[bool, str]:
    """Validate that a command string is safe to execute via shell.

    Returns:
        (is_safe, error_message) tuple. If is_safe is False, error_message explains why.
    """
    if not command:
        return True, ""

    # Check for dangerous shell metacharacters
    found = [c for c in DANGEROUS_CHARS if c in command]
    if found:
        return False, f"Command contains dangerous shell metacharacters: {found}"

    # Check for command chaining patterns
    dangerous_patterns = [
        r"\$\{",  # Variable expansion ${VAR}
        r"\$\(",  # Command substitution $(cmd)
        r"`",     # Backtick command substitution
        r"\|\|",  # OR operator
        r"&&",    # AND operator
        r">>",    # Append redirect
        r"2>",    # Stderr redirect
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            return False, f"Command contains dangerous pattern: {pattern}"

    return True, ""


@dataclass
class PerfCounter:
    """A single perf counter value with multiplexing correction."""
    event: str
    value: float
    unit: str = ""
    enabled_ns: float = 0
    running_ns: float = 0

    @property
    def multiplexing_ratio(self) -> float:
        """Fraction of time this counter was actually counting."""
        if self.enabled_ns <= 0:
            return 0.0
        return self.running_ns / self.enabled_ns

    @property
    def corrected_value(self) -> float:
        """Value corrected for multiplexing."""
        ratio = self.multiplexing_ratio
        if ratio <= 0:
            return 0.0
        return self.value / ratio


@dataclass
class PerfStatResult:
    """Parsed result from a perf stat run."""
    group_name: str
    counters: Dict[str, PerfCounter] = field(default_factory=dict)
    duration_sec: float = 0.0
    raw_output: str = ""

    def get(self, name: str, corrected: bool = True) -> float:
        """Get counter value by logical name."""
        if name not in self.counters:
            return 0.0
        return self.counters[name].corrected_value if corrected else self.counters[name].value


class PerfCollector(BaseCollector):
    """Wraps Linux perf for ARM PMU event collection."""

    def __init__(self, output_dir: Path, kunpeng_model: str = "920"):
        super().__init__(output_dir)
        self.model = get_kunpeng_model(kunpeng_model)
        self._available_events: Optional[set] = None

    def collect(self, command: str = "", pid: Optional[int] = None,
                duration_sec: int = 60, **kwargs) -> CollectionResult:
        """Run full perf stat collection across all event groups."""
        results = self.stat_all_groups(
            command=command, pid=pid, duration_sec=duration_sec
        )
        data = {}
        errors = []
        raw_files = {}
        for group_name, stat_result in results.items():
            data[group_name] = {
                name: counter.corrected_value
                for name, counter in stat_result.counters.items()
            }
            data[f"{group_name}_duration"] = stat_result.duration_sec
            raw_path = self._save_raw(f"perf_stat_{group_name}.json", stat_result.raw_output)
            raw_files[group_name] = raw_path

        return CollectionResult(
            collector_name="perf",
            data=data,
            raw_files=raw_files,
            errors=errors,
        )

    def stat(
        self,
        event_group: EventGroup,
        command: str = "",
        pid: Optional[int] = None,
        duration_sec: Optional[int] = None,
        cpu_list: Optional[str] = None,
        repeat: int = 3,
    ) -> PerfStatResult:
        """Run perf stat for a single event group.

        Uses JSON output (-j) and applies multiplexing correction.
        """
        # Security: validate command before passing to shell
        if command:
            is_safe, error_msg = validate_command_safety(command)
            if not is_safe:
                raise ValueError(f"Unsafe command rejected: {error_msg}")

        event_str = build_perf_event_string(event_group, self.model.pmu_name)
        cmd = ["perf", "stat", "-x", ",", "-e", event_str, "-r", str(repeat)]

        if cpu_list:
            cmd.extend(["-C", cpu_list])

        if pid is not None:
            cmd.extend(["-p", str(pid)])
            if duration_sec:
                cmd.extend(["--", "sleep", str(duration_sec)])
        elif command:
            cmd.extend(["--", "sh", "-c", command])
        elif duration_sec:
            cmd.extend(["-a", "--", "sleep", str(duration_sec)])
        else:
            cmd.extend(["-a", "--", "sleep", "10"])

        result = run_cmd(cmd, timeout_sec=max((duration_sec or 60) * repeat + 60, 300))

        # perf stat outputs CSV to stderr
        raw_output = result.stderr
        counters = self._parse_perf_stat_csv(raw_output, event_group)

        # Extract duration from perf stat output
        duration = self._extract_duration(raw_output, duration_sec or 10)

        return PerfStatResult(
            group_name=event_group.name,
            counters=counters,
            duration_sec=duration,
            raw_output=raw_output,
        )

    def stat_all_groups(
        self,
        command: str = "",
        pid: Optional[int] = None,
        duration_sec: int = 60,
        repeat: int = 3,
    ) -> Dict[str, PerfStatResult]:
        """Run perf stat for ALL core event groups sequentially.

        Sequential collection avoids excessive multiplexing and ensures
        each group gets dedicated use of the 6 programmable PMU counters.
        """
        results = {}
        for group in get_all_core_event_groups():
            self.log.info("Collecting event group: %s", group.name)
            try:
                stat_result = self.stat(
                    event_group=group,
                    command=command,
                    pid=pid,
                    duration_sec=duration_sec,
                    repeat=repeat,
                )
                results[group.name] = stat_result
            except Exception as e:
                self.log.error("Failed to collect %s: %s", group.name, e)
        return results

    def record(
        self,
        command: str = "",
        events: Optional[List[str]] = None,
        frequency: int = 99,
        call_graph: str = "fp",
        duration_sec: Optional[int] = None,
        pid: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> Path:
        """Run perf record for sampling-based profiling."""
        # Security: validate command before passing to shell
        if command:
            is_safe, error_msg = validate_command_safety(command)
            if not is_safe:
                raise ValueError(f"Unsafe command rejected: {error_msg}")

        if output_file is None:
            output_file = str(self.output_dir / "perf.data")

        cmd = ["perf", "record", "-o", output_file, "-F", str(frequency),
               "-g", f"--call-graph={call_graph}"]

        if events:
            cmd.extend(["-e", ",".join(events)])

        if pid is not None:
            cmd.extend(["-p", str(pid)])
            if duration_sec:
                cmd.extend(["--", "sleep", str(duration_sec)])
        elif command:
            cmd.extend(["--", "sh", "-c", command])
        elif duration_sec:
            cmd.extend(["-a", "--", "sleep", str(duration_sec)])

        result = run_cmd(cmd, timeout_sec=max((duration_sec or 60) + 60, 300))
        if not result.ok:
            self.log.error("perf record failed: %s", result.stderr[:500])

        return Path(output_file)

    def topdown_analysis(
        self,
        command: str = "",
        pid: Optional[int] = None,
        duration_sec: int = 30,
    ) -> Dict[str, float]:
        """Run TopDown Level 1 analysis using ARM stall events.

        ARM TopDown L1:
          frontend_bound = stall_frontend / cycles
          backend_bound  = stall_backend / cycles
          retiring       = (instructions / cycles) / dispatch_width
          bad_speculation = 1.0 - frontend - backend - retiring  (clamped)
        """
        stat_result = self.stat(
            CORE_EVENT_GROUPS["topdown_l1"],
            command=command,
            pid=pid,
            duration_sec=duration_sec,
        )

        cycles = stat_result.get("cycles")
        instructions = stat_result.get("instructions")
        stall_fe = stat_result.get("stall_frontend")
        stall_be = stat_result.get("stall_backend")

        if cycles <= 0:
            return {"frontend_bound": 0, "backend_bound": 0,
                    "retiring": 0, "bad_speculation": 0}

        frontend_bound = stall_fe / cycles
        backend_bound = stall_be / cycles
        retiring = (instructions / cycles) / self.model.dispatch_width
        bad_speculation = 1.0 - frontend_bound - backend_bound - retiring

        # Clamp bad_speculation (can go negative due to measurement noise)
        bad_speculation = max(0.0, min(1.0, bad_speculation))

        # Rebalance if total exceeds 1.0
        total = frontend_bound + backend_bound + retiring + bad_speculation
        if total > 0:
            frontend_bound /= total
            backend_bound /= total
            retiring /= total
            bad_speculation /= total

        return {
            "frontend_bound": round(frontend_bound, 4),
            "backend_bound": round(backend_bound, 4),
            "retiring": round(retiring, 4),
            "bad_speculation": round(bad_speculation, 4),
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _parse_perf_stat_csv(
        self, output: str, group: EventGroup
    ) -> Dict[str, PerfCounter]:
        """Parse perf stat CSV output (-x , format).

        CSV columns: value,,event,stddev%,time_enabled_ns,pcnt_running,,comment
        Compatible with perf 5.10+ (does not require -j JSON support).
        """
        counters: Dict[str, PerfCounter] = {}
        event_to_name = {v: k for k, v in group.events.items()}

        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue

            raw_value = parts[0].strip()
            event_raw = parts[2].strip() if len(parts) > 2 else ""
            # parts[3] = stddev% (skip), parts[4] = time_enabled_ns, parts[5] = pcnt_running
            time_enabled = parts[4].strip() if len(parts) > 4 else "0"
            pcnt_running = parts[5].strip() if len(parts) > 5 else "100.00"

            if not event_raw or raw_value in ("<not counted>", "<not supported>", ""):
                continue

            # Normalize event name: "armv8_pmuv3/cpu_cycles/" -> "cpu_cycles"
            event_clean = event_raw.strip()
            m = re.search(r"/([^/]+)/", event_clean)
            if m:
                event_clean = m.group(1)

            logical_name = event_to_name.get(event_clean, event_clean)

            try:
                value = float(raw_value.replace(",", ""))
            except ValueError:
                value = 0.0

            try:
                enabled_ns = float(time_enabled)
                pct = float(pcnt_running) / 100.0
                running_ns = enabled_ns * pct
            except ValueError:
                enabled_ns = 0.0
                running_ns = 0.0

            counters[logical_name] = PerfCounter(
                event=event_raw,
                value=value,
                unit="",
                enabled_ns=enabled_ns,
                running_ns=running_ns,
            )

        return counters

    def _parse_perf_stat_json(
        self, output: str, group: EventGroup
    ) -> Dict[str, PerfCounter]:
        """Parse perf stat JSON output (one JSON object per line).

        Kept for reference; active code now uses _parse_perf_stat_csv.
        """
        counters: Dict[str, PerfCounter] = {}
        event_to_name = {v: k for k, v in group.events.items()}

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_raw = obj.get("event", "")
            counter_value = obj.get("counter-value", "0")
            unit = obj.get("unit", "")

            # Clean up event name to match our definitions
            event_clean = event_raw.strip()
            # Remove PMU prefix: "armv8_pmuv3/cpu_cycles/" -> "cpu_cycles"
            m = re.search(r"/([^/]+)/", event_clean)
            if m:
                event_clean = m.group(1)

            # Map to logical name
            logical_name = event_to_name.get(event_clean, event_clean)

            try:
                value = float(str(counter_value).replace(",", ""))
            except ValueError:
                value = 0.0

            counters[logical_name] = PerfCounter(
                event=event_raw,
                value=value,
                unit=unit,
                enabled_ns=float(obj.get("event-runtime", 0)),
                running_ns=float(obj.get("pcnt-running", 100)) / 100.0
                           * float(obj.get("event-runtime", 0)),
            )

        return counters

    def _extract_duration(self, raw_output: str, default: float) -> float:
        """Extract elapsed time from perf stat output."""
        m = re.search(r"([\d.]+)\s+seconds time elapsed", raw_output)
        if m:
            return float(m.group(1))
        return default

    def detect_available_events(self) -> set:
        """Probe which PMU events are available on current hardware."""
        if self._available_events is not None:
            return self._available_events

        self._available_events = set()
        for group in get_all_core_event_groups():
            for name, event in group.events.items():
                event_str = f"{self.model.pmu_name}/{event}/"
                result = run_cmd(
                    ["perf", "stat", "-e", event_str, "true"],
                    timeout_sec=10,
                )
                if result.ok and "not supported" not in result.stderr:
                    self._available_events.add(name)

        self.log.info("Detected %d available PMU events", len(self._available_events))
        return self._available_events

    # -----------------------------------------------------------------------
    # Uncore PMU collection (DDR bandwidth, L3 cache)
    # -----------------------------------------------------------------------

    def collect_uncore(
        self,
        duration_sec: int = 30,
        sccl_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Collect uncore PMU events for DDR bandwidth and L3 cache stats.

        Kunpeng uncore PMUs provide:
        - DDR controller read/write counts (flux_rd, flux_wr)
        - L3 cache hit/miss from uncore perspective

        Args:
            duration_sec: Collection duration
            sccl_ids: SCCL (Super CPU Cluster) IDs to monitor.
                      Default: [1, 3, 5, 7] for 4-socket Kunpeng 920

        Returns:
            Dict with 'ddr_bandwidth' and 'l3_cache_uncore' data
        """
        result: Dict[str, Any] = {}

        # Collect DDR bandwidth
        ddr_data = self._collect_ddr_bandwidth(duration_sec, sccl_ids)
        if ddr_data:
            result["ddr_bandwidth"] = ddr_data

        # Collect L3 uncore stats
        l3_data = self._collect_l3_uncore(duration_sec, sccl_ids)
        if l3_data:
            result["l3_cache_uncore"] = l3_data

        return result

    def _collect_ddr_bandwidth(
        self,
        duration_sec: int,
        sccl_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Collect DDR controller bandwidth using uncore PMU.

        Kunpeng DDR controller events:
        - flux_rd: Number of read beats (32 bytes each)
        - flux_wr: Number of write beats (32 bytes each)
        """
        if sccl_ids is None:
            sccl_ids = [1, 3, 5, 7]  # Default 4-socket

        # Resolve uncore events for all DDR controllers
        group = UNCORE_EVENT_GROUPS["ddr_bandwidth"]
        events = resolve_uncore_events(group, sccl_ids, unit_ids=[0, 1, 2, 3])

        # Build perf command
        event_str = ",".join(events)
        cmd = [
            "perf", "stat", "-x", ",", "-e", event_str,
            "-a", "--", "sleep", str(duration_sec),
        ]

        result = run_cmd(cmd, timeout_sec=duration_sec + 60)
        if not result.ok:
            self.log.warning("Uncore DDR collection failed: %s", result.stderr[:200])
            return {}

        # Parse results
        total_read_beats = 0
        total_write_beats = 0

        for line in result.stderr.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue

            try:
                value = float(parts[0].replace(",", ""))
            except ValueError:
                continue

            event_name = parts[2].strip() if len(parts) > 2 else ""
            if "flux_rd" in event_name:
                total_read_beats += value
            elif "flux_wr" in event_name:
                total_write_beats += value

        # Convert beats to bandwidth (each beat = 32 bytes)
        bytes_per_beat = 32
        read_bytes = total_read_beats * bytes_per_beat
        write_bytes = total_write_beats * bytes_per_beat

        read_gbps = read_bytes / (duration_sec * 1e9)
        write_gbps = write_bytes / (duration_sec * 1e9)
        total_gbps = read_gbps + write_gbps

        # Calculate utilization
        max_bw = self.model.max_bandwidth_gbps
        utilization = total_gbps / max_bw if max_bw > 0 else 0.0

        return {
            "read_gbps": round(read_gbps, 2),
            "write_gbps": round(write_gbps, 2),
            "total_gbps": round(total_gbps, 2),
            "utilization": round(utilization, 4),
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "duration_sec": duration_sec,
        }

    def _collect_l3_uncore(
        self,
        duration_sec: int,
        sccl_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Collect L3 cache statistics from uncore PMU.

        Kunpeng L3 uncore events:
        - rd_hit_cpipe: L3 read hits
        - rd_miss_cpipe: L3 read misses
        - wr_hit_cpipe: L3 write hits
        - wr_miss_cpipe: L3 write misses
        """
        if sccl_ids is None:
            sccl_ids = [1, 3, 5, 7]

        group = UNCORE_EVENT_GROUPS["l3_cache_uncore"]
        events = resolve_uncore_events(group, sccl_ids, unit_ids=[0, 1, 2, 3])

        event_str = ",".join(events)
        cmd = [
            "perf", "stat", "-x", ",", "-e", event_str,
            "-a", "--", "sleep", str(duration_sec),
        ]

        result = run_cmd(cmd, timeout_sec=duration_sec + 60)
        if not result.ok:
            self.log.warning("Uncore L3 collection failed: %s", result.stderr[:200])
            return {}

        # Parse results
        rd_hit = 0
        rd_miss = 0
        wr_hit = 0
        wr_miss = 0

        for line in result.stderr.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue

            try:
                value = float(parts[0].replace(",", ""))
            except ValueError:
                continue

            event_name = parts[2].strip() if len(parts) > 2 else ""
            if "rd_hit_cpipe" in event_name:
                rd_hit += value
            elif "rd_miss_cpipe" in event_name:
                rd_miss += value
            elif "wr_hit_cpipe" in event_name:
                wr_hit += value
            elif "wr_miss_cpipe" in event_name:
                wr_miss += value

        total_access = rd_hit + rd_miss + wr_hit + wr_miss
        total_hits = rd_hit + wr_hit
        total_misses = rd_miss + wr_miss

        hit_rate = total_hits / total_access if total_access > 0 else 0.0
        miss_rate = total_misses / total_access if total_access > 0 else 0.0

        return {
            "rd_hit": rd_hit,
            "rd_miss": rd_miss,
            "wr_hit": wr_hit,
            "wr_miss": wr_miss,
            "hit_rate": round(hit_rate, 4),
            "miss_rate": round(miss_rate, 4),
            "total_access": total_access,
        }
