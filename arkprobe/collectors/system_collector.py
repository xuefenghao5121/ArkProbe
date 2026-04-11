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

        try:
            data["power_thermal"] = self.collect_power_thermal(duration_sec)
        except Exception as e:
            errors.append(f"power_thermal: {e}")

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
    # Power and thermal metrics
    # -------------------------------------------------------------------

    def collect_power_thermal(self, duration_sec: int = 10) -> Dict[str, Any]:
        """Collect power consumption, temperature, and C-state residency.

        Data sources:
        - /sys/class/hwmon (hardware monitoring sensors)
        - /sys/class/thermal (thermal zones)
        - /sys/devices/system/cpu/cpu*/cpuidle (C-state statistics)
        - /sys/devices/system/cpu/cpu*/cpufreq (frequency stats)
        """
        result: Dict[str, Any] = {}

        # Power and temperature from hwmon
        hwmon_data = self._collect_hwmon()
        result.update(hwmon_data)

        # Thermal zones
        thermal_data = self._collect_thermal_zones()
        result.update(thermal_data)

        # C-state residency
        cstate_data = self._collect_cstate_residency(duration_sec)
        result.update(cstate_data)

        # Frequency stats
        freq_data = self._collect_frequency_stats()
        result.update(freq_data)

        return result

    def _collect_hwmon(self) -> Dict[str, Any]:
        """Collect from /sys/class/hwmon (hardware monitoring sensors)."""
        result: Dict[str, Any] = {}
        hwmon_base = Path("/sys/class/hwmon")

        if not hwmon_base.exists():
            return result

        for hwmon_dir in sorted(hwmon_base.iterdir()):
            if not hwmon_dir.is_dir():
                continue

            # Get device name
            name_path = hwmon_dir / "name"
            if not name_path.exists():
                continue
            device_name = name_path.read_text().strip().lower()

            # Look for power sensors (power*_input, in microwatts)
            for power_file in hwmon_dir.glob("power*_input"):
                try:
                    power_uw = int(power_file.read_text().strip())
                    power_w = power_uw / 1_000_000.0

                    # Classify by device name or sensor name
                    sensor_name = power_file.stem.replace("_input", "")
                    if "cpu" in device_name or "cpu" in sensor_name:
                        if result.get("cpu_power_w") is None:
                            result["cpu_power_w"] = power_w
                    elif "dram" in device_name or "mem" in sensor_name:
                        if result.get("dram_power_w") is None:
                            result["dram_power_w"] = power_w
                    elif "gpu" in device_name:
                        if result.get("gpu_power_w") is None:
                            result["gpu_power_w"] = power_w
                    else:
                        # Total or other
                        if result.get("total_power_w") is None:
                            result["total_power_w"] = power_w
                except (OSError, ValueError):
                    pass

            # Look for temperature sensors (temp*_input, in millidegrees)
            for temp_file in hwmon_dir.glob("temp*_input"):
                try:
                    temp_mc = int(temp_file.read_text().strip())
                    temp_c = temp_mc / 1000.0

                    sensor_name = temp_file.stem.replace("_input", "")
                    if "cpu" in device_name or "cpu" in sensor_name or "core" in sensor_name:
                        if result.get("cpu_temp_c") is None:
                            result["cpu_temp_c"] = temp_c
                    elif "dram" in device_name or "mem" in sensor_name:
                        if result.get("dram_temp_c") is None:
                            result["dram_temp_c"] = temp_c
                    elif "mobo" in device_name or "board" in sensor_name or "ambient" in sensor_name:
                        if result.get("motherboard_temp_c") is None:
                            result["motherboard_temp_c"] = temp_c
                except (OSError, ValueError):
                    pass

            # Look for max temperature thresholds (temp*_max)
            for temp_max_file in hwmon_dir.glob("temp*_max"):
                try:
                    temp_mc = int(temp_max_file.read_text().strip())
                    temp_c = temp_mc / 1000.0
                    sensor_name = temp_max_file.stem.replace("_max", "")
                    if "cpu" in device_name or "cpu" in sensor_name:
                        if result.get("cpu_temp_max_c") is None:
                            result["cpu_temp_max_c"] = temp_c
                except (OSError, ValueError):
                    pass

        return result

    def _collect_thermal_zones(self) -> Dict[str, Any]:
        """Collect from /sys/class/thermal (thermal zones)."""
        result: Dict[str, Any] = {}
        thermal_base = Path("/sys/class/thermal")

        if not thermal_base.exists():
            return result

        for zone_dir in sorted(thermal_base.iterdir()):
            if not zone_dir.name.startswith("thermal_zone"):
                continue

            try:
                type_path = zone_dir / "type"
                if not type_path.exists():
                    continue
                zone_type = type_path.read_text().strip().lower()

                temp_path = zone_dir / "temp"
                if temp_path.exists():
                    temp_mc = int(temp_path.read_text().strip())
                    temp_c = temp_mc / 1000.0

                    # Map zone type to result field
                    if "cpu" in zone_type or "core" in zone_type:
                        if result.get("cpu_temp_c") is None:
                            result["cpu_temp_c"] = temp_c
                    elif "dram" in zone_type or "mem" in zone_type:
                        if result.get("dram_temp_c") is None:
                            result["dram_temp_c"] = temp_c
                    elif "mobo" in zone_type or "board" in zone_type:
                        if result.get("motherboard_temp_c") is None:
                            result["motherboard_temp_c"] = temp_c
            except (OSError, ValueError):
                pass

        return result

    def _collect_cstate_residency(self, duration_sec: int) -> Dict[str, Any]:
        """Collect C-state residency from /sys/devices/system/cpu/cpu*/cpuidle."""
        result: Dict[str, Any] = {}
        cpu_base = Path("/sys/devices/system/cpu")

        if not cpu_base.exists():
            return result

        # Collect initial C-state times
        before = self._read_cstate_times()
        time.sleep(duration_sec)
        after = self._read_cstate_times()

        # Calculate residency percentages
        total_time = 0
        for state in before:
            delta = after.get(state, 0) - before.get(state, 0)
            if delta > 0:
                total_time += delta

        if total_time > 0:
            for state in before:
                delta = after.get(state, 0) - before.get(state, 0)
                residency = delta / total_time if delta > 0 else 0.0

                # Map state name to result field
                state_lower = state.lower()
                if "c0" in state_lower or state_lower == "poll":
                    result["c0_residency"] = round(residency, 4)
                elif "c1" in state_lower:
                    result["c1_residency"] = round(residency, 4)
                elif "c2" in state_lower:
                    result["c2_residency"] = round(residency, 4)
                elif "c3" in state_lower:
                    result["c3_residency"] = round(residency, 4)
                elif "c6" in state_lower or "c7" in state_lower:
                    result["c6_residency"] = round(residency, 4)

        return result

    def _read_cstate_times(self) -> Dict[str, int]:
        """Read current C-state times (in microseconds) for all CPUs."""
        cstate_times: Dict[str, int] = {}
        cpu_base = Path("/sys/devices/system/cpu")

        for cpu_dir in sorted(cpu_base.iterdir()):
            if not cpu_dir.name.startswith("cpu"):
                continue
            if not cpu_dir.name[3:].isdigit():
                continue

            cpuidle_dir = cpu_dir / "cpuidle"
            if not cpuidle_dir.exists():
                continue

            for state_dir in sorted(cpuidle_dir.iterdir()):
                if not state_dir.name.startswith("state"):
                    continue

                name_path = state_dir / "name"
                time_path = state_dir / "time"

                if name_path.exists() and time_path.exists():
                    try:
                        state_name = name_path.read_text().strip()
                        time_us = int(time_path.read_text().strip())
                        # Aggregate across all CPUs
                        cstate_times[state_name] = cstate_times.get(state_name, 0) + time_us
                    except (OSError, ValueError):
                        pass

        return cstate_times

    def _collect_frequency_stats(self) -> Dict[str, Any]:
        """Collect CPU frequency statistics."""
        result: Dict[str, Any] = {}
        cpu_base = Path("/sys/devices/system/cpu")

        if not cpu_base.exists():
            return result

        freqs = []
        for cpu_dir in sorted(cpu_base.iterdir()):
            if not cpu_dir.name.startswith("cpu"):
                continue
            if not cpu_dir.name[3:].isdigit():
                continue

            cpufreq_dir = cpu_dir / "cpufreq"
            if not cpufreq_dir.exists():
                continue

            # Current frequency
            cur_freq_path = cpufreq_dir / "scaling_cur_freq"
            if cur_freq_path.exists():
                try:
                    freq_khz = int(cur_freq_path.read_text().strip())
                    freqs.append(freq_khz / 1000.0)  # Convert to MHz
                except (OSError, ValueError):
                    pass

        if freqs:
            result["avg_freq_mhz"] = round(sum(freqs) / len(freqs), 1)
            result["min_freq_mhz"] = round(min(freqs), 1)
            result["max_freq_mhz"] = round(max(freqs), 1)

        # Check for thermal throttling via thermal_zone
        thermal_base = Path("/sys/class/thermal")
        if thermal_base.exists():
            for zone_dir in thermal_base.iterdir():
                if not zone_dir.name.startswith("thermal_zone"):
                    continue
                # Check for throttling indicator
                # Some systems expose throttling state
                throttle_path = zone_dir / "cdev0_cur_state"
                if throttle_path.exists():
                    try:
                        state = int(throttle_path.read_text().strip())
                        if state > 0:
                            result["thermal_throttling_pct"] = min(100.0, state * 10.0)
                    except (OSError, ValueError):
                        pass

        return result

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
