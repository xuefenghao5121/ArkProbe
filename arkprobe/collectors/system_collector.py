"""System-level metrics collection from /proc, /sys, and standard tools."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseCollector, CollectionResult
from ..utils.platform_detect import PlatformInfo, detect_platform
from ..utils.process import run_cmd


class SystemCollector(BaseCollector):
    """Collect system-level metrics from /proc, /sys, and standard tools."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self._platform: Optional[PlatformInfo] = None

    def collect(self, duration_sec: int = 10, **kwargs) -> CollectionResult:
        """Collect all system metrics."""
        data: Dict[str, Any] = {}
        errors = []

        try:
            data["platform"] = self.collect_platform_info()
        except Exception as e:
            errors.append(f"platform: {e}")

        try:
            data["cpu_utilization"] = self.collect_cpu_utilization(duration_sec)
        except Exception as e:
            errors.append(f"cpu_util: {e}")

        try:
            data["memory"] = self.collect_memory_info()
        except Exception as e:
            errors.append(f"memory: {e}")

        try:
            data["numa"] = self.collect_numa_stats()
        except Exception as e:
            errors.append(f"numa: {e}")

        try:
            data["disk"] = self.collect_disk_stats(duration_sec)
        except Exception as e:
            errors.append(f"disk: {e}")

        raw_path = self._save_raw(
            "system_metrics.json",
            json.dumps(data, indent=2, default=str),
        )

        return CollectionResult(
            collector_name="system",
            data=data,
            raw_files={"system": raw_path},
            errors=errors,
        )

    def collect_platform_info(self) -> Dict[str, Any]:
        """Detect platform and return as dict."""
        if self._platform is None:
            self._platform = detect_platform()
        p = self._platform
        return {
            "arch": p.arch,
            "kernel_version": p.kernel_version,
            "kunpeng_model": p.kunpeng_model,
            "cpu_model_name": p.cpu_model_name,
            "socket_count": p.socket_count,
            "cores_per_socket": p.cores_per_socket,
            "threads_per_core": p.threads_per_core,
            "total_cores": p.total_cores,
            "numa_nodes": p.numa_nodes,
            "l1d_cache_kb": p.l1d_cache_kb,
            "l1i_cache_kb": p.l1i_cache_kb,
            "l2_cache_kb": p.l2_cache_kb,
            "l3_cache_mb": p.l3_cache_mb,
        }

    def collect_memory_info(self) -> Dict[str, Any]:
        """Parse /proc/meminfo."""
        info = {}
        try:
            content = Path("/proc/meminfo").read_text()
            for line in content.splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_parts = parts[1].strip().split()
                    if val_parts:
                        try:
                            info[key] = int(val_parts[0])  # value in kB
                        except ValueError:
                            info[key] = val_parts[0]
        except Exception as e:
            self.log.error("Failed to read /proc/meminfo: %s", e)

        return {
            "total_kb": info.get("MemTotal", 0),
            "available_kb": info.get("MemAvailable", 0),
            "free_kb": info.get("MemFree", 0),
            "buffers_kb": info.get("Buffers", 0),
            "cached_kb": info.get("Cached", 0),
            "hugepages_total": info.get("HugePages_Total", 0),
            "hugepages_free": info.get("HugePages_Free", 0),
            "hugepage_size_kb": info.get("Hugepagesize", 2048),
        }

    def collect_numa_stats(self) -> Dict[str, Any]:
        """Collect NUMA statistics from /sys and numastat."""
        numa_data: Dict[str, Any] = {"nodes": {}}
        numa_base = Path("/sys/devices/system/node")
        if not numa_base.exists():
            return numa_data

        for node_dir in sorted(numa_base.iterdir()):
            if not node_dir.name.startswith("node"):
                continue
            node_id = node_dir.name
            node_info: Dict[str, int] = {}

            numastat = node_dir / "numastat"
            if numastat.exists():
                for line in numastat.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            node_info[parts[0]] = int(parts[1])
                        except ValueError:
                            pass

            meminfo = node_dir / "meminfo"
            if meminfo.exists():
                for line in meminfo.read_text().splitlines():
                    m = re.search(r"MemTotal:\s+(\d+)", line)
                    if m:
                        node_info["mem_total_kb"] = int(m.group(1))
                    m = re.search(r"MemFree:\s+(\d+)", line)
                    if m:
                        node_info["mem_free_kb"] = int(m.group(1))

            numa_data["nodes"][node_id] = node_info

        # Calculate local access ratio
        total_local = sum(
            n.get("local_node", 0) for n in numa_data["nodes"].values()
        )
        total_other = sum(
            n.get("other_node", 0) for n in numa_data["nodes"].values()
        )
        total = total_local + total_other
        numa_data["local_access_ratio"] = total_local / total if total > 0 else 1.0

        return numa_data

    def collect_cpu_utilization(
        self, duration_sec: int = 10
    ) -> Dict[str, Any]:
        """Sample /proc/stat to get CPU utilization breakdown."""

        def read_stat():
            content = Path("/proc/stat").read_text()
            result = {}
            for line in content.splitlines():
                if line.startswith("cpu"):
                    parts = line.split()
                    name = parts[0]
                    values = [int(v) for v in parts[1:]]
                    # user, nice, system, idle, iowait, irq, softirq, steal
                    result[name] = {
                        "user": values[0] + values[1],
                        "system": values[2],
                        "idle": values[3],
                        "iowait": values[4] if len(values) > 4 else 0,
                        "irq": values[5] + values[6] if len(values) > 6 else 0,
                    }
            return result

        before = read_stat()
        time.sleep(duration_sec)
        after = read_stat()

        # Compute deltas for aggregate CPU
        cpu_before = before.get("cpu", {})
        cpu_after = after.get("cpu", {})
        deltas = {k: cpu_after.get(k, 0) - cpu_before.get(k, 0) for k in cpu_after}
        total = sum(deltas.values())

        return {
            "user_pct": (deltas.get("user", 0) / total * 100) if total > 0 else 0,
            "system_pct": (deltas.get("system", 0) / total * 100) if total > 0 else 0,
            "idle_pct": (deltas.get("idle", 0) / total * 100) if total > 0 else 0,
            "iowait_pct": (deltas.get("iowait", 0) / total * 100) if total > 0 else 0,
            "irq_pct": (deltas.get("irq", 0) / total * 100) if total > 0 else 0,
            "total_utilization_pct": 100 - (deltas.get("idle", 0) / total * 100 if total > 0 else 0),
        }

    def collect_disk_stats(self, duration_sec: int = 10) -> Dict[str, Any]:
        """Parse /proc/diskstats for I/O metrics."""

        def read_diskstats():
            stats = {}
            content = Path("/proc/diskstats").read_text()
            for line in content.splitlines():
                parts = line.split()
                if len(parts) >= 14:
                    name = parts[2]
                    # Skip partitions (only major block devices)
                    if re.match(r"^(sd[a-z]|vd[a-z]|nvme\d+n\d+)$", name) or name.startswith("dm-"):
                        stats[name] = {
                            "reads": int(parts[3]),
                            "read_sectors": int(parts[5]),
                            "read_time_ms": int(parts[6]),
                            "writes": int(parts[7]),
                            "write_sectors": int(parts[9]),
                            "write_time_ms": int(parts[10]),
                            "io_in_progress": int(parts[11]),
                            "io_time_ms": int(parts[12]),
                        }
            return stats

        before = read_diskstats()
        time.sleep(duration_sec)
        after = read_diskstats()

        result = {}
        for dev in after:
            if dev not in before:
                continue
            delta = {k: after[dev][k] - before[dev].get(k, 0) for k in after[dev]}
            sector_size = 512  # bytes
            result[dev] = {
                "iops_read": delta["reads"] / duration_sec,
                "iops_write": delta["writes"] / duration_sec,
                "throughput_read_mbps": (delta["read_sectors"] * sector_size)
                                       / (duration_sec * 1e6),
                "throughput_write_mbps": (delta["write_sectors"] * sector_size)
                                        / (duration_sec * 1e6),
                "avg_read_latency_ms": delta["read_time_ms"] / max(delta["reads"], 1),
                "avg_write_latency_ms": delta["write_time_ms"] / max(delta["writes"], 1),
                "utilization_pct": delta["io_time_ms"] / (duration_sec * 1000) * 100,
            }

        return result

    def collect_context_switches(self, duration_sec: int = 10) -> Dict[str, Any]:
        """Measure context switch rate from /proc/stat."""
        def read_cs():
            content = Path("/proc/stat").read_text()
            for line in content.splitlines():
                if line.startswith("ctxt"):
                    return int(line.split()[1])
            return 0

        before = read_cs()
        time.sleep(duration_sec)
        after = read_cs()

        return {
            "context_switches_per_sec": (after - before) / duration_sec,
        }
