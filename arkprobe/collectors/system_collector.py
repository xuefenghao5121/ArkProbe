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

        try:
            data["platform_config"] = self.collect_platform_config()
        except Exception as e:
            errors.append(f"platform_config: {e}")

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

    # -------------------------------------------------------------------
    # Platform configuration snapshot (for optimization analysis)
    # -------------------------------------------------------------------

    def collect_platform_config(self) -> Dict[str, Any]:
        """Collect the full platform tuning configuration snapshot."""
        return {
            "os": self.collect_os_config(),
            "bios": self.collect_bios_config(),
            "driver": self.collect_driver_config(),
        }

    def collect_os_config(self) -> Dict[str, Any]:
        """Read OS-level tuning parameters from /proc/sys, /sys."""
        cfg: Dict[str, Any] = {}

        # Hugepages
        cfg["hugepages_total"] = self._read_int("/proc/sys/vm/nr_hugepages", 0)
        cfg["hugepage_size_kb"] = self._read_int(
            "/proc/meminfo", 2048, pattern=r"Hugepagesize:\s+(\d+)")

        # Transparent Huge Pages
        thp_path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
        if thp_path.exists():
            content = thp_path.read_text().strip()
            m = re.search(r"\[(\w+)\]", content)
            cfg["transparent_hugepage"] = m.group(1) if m else "unknown"
        else:
            cfg["transparent_hugepage"] = "unknown"

        # CPU governor
        gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        cfg["cpu_governor"] = gov_path.read_text().strip() if gov_path.exists() else "unknown"

        # VM parameters
        cfg["swappiness"] = self._read_int("/proc/sys/vm/swappiness", 60)
        cfg["dirty_ratio"] = self._read_int("/proc/sys/vm/dirty_ratio", 20)
        cfg["dirty_background_ratio"] = self._read_int(
            "/proc/sys/vm/dirty_background_ratio", 10)

        # NUMA balancing
        cfg["numa_balancing"] = self._read_int(
            "/proc/sys/kernel/numa_balancing", 1) == 1

        # Network params
        cfg["netdev_max_backlog"] = self._read_int(
            "/proc/sys/net/core/netdev_max_backlog", 1000)
        cfg["somaxconn"] = self._read_int(
            "/proc/sys/net/core/somaxconn", 4096)
        cfg["tcp_max_syn_backlog"] = self._read_int(
            "/proc/sys/net/ipv4/tcp_max_syn_backlog", 1024)

        # IO schedulers per block device
        io_scheds: Dict[str, str] = {}
        for sched_path in Path("/sys/block").glob("*/queue/scheduler"):
            dev = sched_path.parts[-3]
            content = sched_path.read_text().strip()
            m = re.search(r"\[(\w[\w-]*)\]", content)
            if m:
                io_scheds[dev] = m.group(1)
        cfg["io_schedulers"] = io_scheds

        # Scheduler tuning
        gran_path = "/proc/sys/kernel/sched_min_granularity_ns"
        if Path(gran_path).exists():
            cfg["sched_min_granularity_ns"] = self._read_int(gran_path, 0)
        mig_path = "/proc/sys/kernel/sched_migration_cost_ns"
        if Path(mig_path).exists():
            cfg["sched_migration_cost_ns"] = self._read_int(mig_path, 0)

        return cfg

    def collect_bios_config(self) -> Dict[str, Any]:
        """Detect BIOS-level settings from /sys (best-effort)."""
        cfg: Dict[str, Any] = {}

        # SMT: threads_per_core > 1 means enabled
        platform = self.collect_platform_info()
        cfg["smt_enabled"] = platform.get("threads_per_core", 1) > 1

        # NUMA: multiple nodes implies NUMA enabled
        cfg["numa_enabled"] = platform.get("numa_nodes", 1) > 1

        # Power / energy performance preference
        epp_path = Path(
            "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
        if epp_path.exists():
            cfg["power_profile"] = epp_path.read_text().strip()
        else:
            gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            cfg["power_profile"] = gov.read_text().strip() if gov.exists() else "unknown"

        # C-states: check if any C-state > C0 is enabled
        c_states_enabled = None
        idle_base = Path("/sys/devices/system/cpu/cpu0/cpuidle")
        if idle_base.exists():
            c_states_enabled = False
            for state_dir in sorted(idle_base.iterdir()):
                if not state_dir.name.startswith("state"):
                    continue
                disable_path = state_dir / "disable"
                if disable_path.exists():
                    disabled = disable_path.read_text().strip() == "1"
                    if not disabled and state_dir.name != "state0":
                        c_states_enabled = True
                        break
        cfg["c_states_enabled"] = c_states_enabled

        # HW Prefetcher — Kunpeng 920 may expose at this path
        prefetch_path = Path("/sys/devices/system/cpu/cpu0/prefetch")
        if prefetch_path.exists():
            val = prefetch_path.read_text().strip()
            cfg["hw_prefetcher_enabled"] = val not in ("0", "disabled")
        else:
            cfg["hw_prefetcher_enabled"] = None  # not detectable

        # Turbo boost
        boost_path = Path("/sys/devices/system/cpu/cpufreq/boost")
        if boost_path.exists():
            cfg["turbo_boost_enabled"] = boost_path.read_text().strip() == "1"
        else:
            cfg["turbo_boost_enabled"] = None

        return cfg

    def collect_driver_config(self) -> Dict[str, Any]:
        """Collect NIC offload, ring buffer, IRQ, and mount option settings."""
        cfg: Dict[str, Any] = {}

        # NIC offloads and ring buffers
        nic_offloads: Dict[str, Dict[str, bool]] = {}
        nic_ring_buffers: Dict[str, Dict[str, int]] = {}
        net_base = Path("/sys/class/net")
        if net_base.exists():
            for iface_path in net_base.iterdir():
                iface = iface_path.name
                if iface == "lo":
                    continue
                # Check if it's a physical device (has /device link)
                if not (iface_path / "device").exists():
                    continue

                # Offloads via ethtool -k
                offload_result = run_cmd(
                    ["ethtool", "-k", iface], timeout_sec=5)
                if offload_result.ok:
                    offloads = {}
                    for line in offload_result.stdout.splitlines():
                        for feat in ("tcp-segmentation-offload",
                                     "generic-receive-offload",
                                     "large-receive-offload",
                                     "generic-segmentation-offload"):
                            if line.strip().startswith(feat):
                                offloads[feat.replace("-", "_")] = ": on" in line
                    if offloads:
                        nic_offloads[iface] = offloads

                # Ring buffers via ethtool -g
                ring_result = run_cmd(
                    ["ethtool", "-g", iface], timeout_sec=5)
                if ring_result.ok:
                    rings = self._parse_ring_buffer(ring_result.stdout)
                    if rings:
                        nic_ring_buffers[iface] = rings

        cfg["nic_offloads"] = nic_offloads
        cfg["nic_ring_buffers"] = nic_ring_buffers

        # irqbalance
        irq_result = run_cmd(
            ["systemctl", "is-active", "irqbalance"], timeout_sec=5)
        cfg["irqbalance_active"] = irq_result.stdout.strip() == "active"

        # Mount options for non-virtual filesystems
        mount_opts: Dict[str, List[str]] = {}
        try:
            mounts = Path("/proc/mounts").read_text()
            for line in mounts.splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    dev, mountpoint, fstype, opts = parts[0], parts[1], parts[2], parts[3]
                    if fstype in ("ext4", "xfs", "btrfs", "ext3"):
                        mount_opts[mountpoint] = opts.split(",")
        except Exception:
            pass
        cfg["mount_options"] = mount_opts

        return cfg

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _read_int(path: str, default: int, pattern: str | None = None) -> int:
        """Read an integer from a file, optionally matching a regex pattern."""
        try:
            content = Path(path).read_text()
            if pattern:
                m = re.search(pattern, content)
                return int(m.group(1)) if m else default
            return int(content.strip())
        except (OSError, ValueError):
            return default

    @staticmethod
    def _parse_ring_buffer(output: str) -> Dict[str, int]:
        """Parse ethtool -g output for current and max RX/TX values."""
        result: Dict[str, int] = {}
        section = ""
        for line in output.splitlines():
            if "Pre-set maximums" in line:
                section = "max"
            elif "Current hardware settings" in line:
                section = "cur"
            elif ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()
                try:
                    v = int(val)
                except ValueError:
                    continue
                if key == "rx" and section == "max":
                    result["rx_max"] = v
                elif key == "rx" and section == "cur":
                    result["rx"] = v
                elif key == "tx" and section == "max":
                    result["tx_max"] = v
                elif key == "tx" and section == "cur":
                    result["tx"] = v
        return result
