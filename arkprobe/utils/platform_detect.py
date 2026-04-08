"""Platform detection for Kunpeng ARM processors."""
import re
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PlatformInfo:
    arch: str  # e.g. "aarch64", "x86_64"
    kernel_version: str
    kunpeng_model: Optional[str]  # "920", "930", or None
    cpu_model_name: str
    socket_count: int
    cores_per_socket: int
    threads_per_core: int
    total_cores: int
    numa_nodes: int
    l1d_cache_kb: int
    l1i_cache_kb: int
    l2_cache_kb: int
    l3_cache_mb: int

def detect_platform() -> PlatformInfo:
    """Auto-detect platform by reading /proc/cpuinfo, lscpu output, /sys/devices/system/cpu."""
    # Read arch
    arch = platform.machine()
    kernel_version = platform.release()

    # Parse /proc/cpuinfo for model name
    cpu_model = "Unknown"
    cpuinfo = Path("/proc/cpuinfo").read_text()
    for line in cpuinfo.splitlines():
        if "model name" in line.lower() or "CPU implementer" in line:
            cpu_model = line.split(":")[-1].strip()
            break

    # Detect Kunpeng model from CPU implementer (0x48 = HiSilicon) and part number
    kunpeng_model = None
    implementer = None
    part = None
    for line in cpuinfo.splitlines():
        if "CPU implementer" in line:
            implementer = line.split(":")[-1].strip()
        if "CPU part" in line:
            part = line.split(":")[-1].strip()

    if implementer == "0x48":  # HiSilicon
        if part == "0xd01":
            kunpeng_model = "920"
        elif part == "0xd02":
            kunpeng_model = "930"

    # Count CPUs from /sys
    cpu_base = Path("/sys/devices/system/cpu")
    online_cpus = 0
    if (cpu_base / "online").exists():
        online_str = (cpu_base / "online").read_text().strip()
        # Parse ranges like "0-63"
        for part_range in online_str.split(","):
            if "-" in part_range:
                lo, hi = part_range.split("-")
                online_cpus += int(hi) - int(lo) + 1
            else:
                online_cpus += 1

    # Detect topology
    socket_count = 1
    cores_per_socket = online_cpus
    threads_per_core = 1

    try:
        # Use first CPU's topology as reference
        topo = cpu_base / "cpu0" / "topology"
        if (topo / "physical_package_id").exists():
            pkg_ids = set()
            core_ids = set()
            for i in range(online_cpus):
                pkg_file = cpu_base / f"cpu{i}" / "topology" / "physical_package_id"
                core_file = cpu_base / f"cpu{i}" / "topology" / "core_id"
                if pkg_file.exists():
                    pkg_ids.add(pkg_file.read_text().strip())
                if core_file.exists():
                    core_ids.add(core_file.read_text().strip())
            socket_count = max(len(pkg_ids), 1)
            physical_cores = max(len(core_ids), 1) * socket_count
            cores_per_socket = physical_cores // socket_count
            threads_per_core = online_cpus // physical_cores if physical_cores > 0 else 1
    except Exception:
        pass

    # NUMA nodes
    numa_nodes = 0
    numa_base = Path("/sys/devices/system/node")
    if numa_base.exists():
        numa_nodes = len([d for d in numa_base.iterdir() if d.name.startswith("node")])
    numa_nodes = max(numa_nodes, 1)

    # Cache sizes - read from sysfs
    l1d_kb, l1i_kb, l2_kb, l3_mb = 64, 64, 512, 32  # defaults for Kunpeng 920
    try:
        cache_base = cpu_base / "cpu0" / "cache"
        if cache_base.exists():
            for idx_dir in sorted(cache_base.iterdir()):
                if not idx_dir.name.startswith("index"):
                    continue
                level = int((idx_dir / "level").read_text().strip())
                ctype = (idx_dir / "type").read_text().strip()
                size_str = (idx_dir / "size").read_text().strip()
                # Parse "64K", "512K", "32768K"
                m = re.match(r"(\d+)([KMG])", size_str)
                if m:
                    val = int(m.group(1))
                    unit = m.group(2)
                    if unit == "M":
                        val *= 1024
                    elif unit == "G":
                        val *= 1024 * 1024
                    # val is now in KB
                    if level == 1 and "Data" in ctype:
                        l1d_kb = val
                    elif level == 1 and "Instruction" in ctype:
                        l1i_kb = val
                    elif level == 2:
                        l2_kb = val
                    elif level == 3:
                        l3_mb = val // 1024
    except Exception:
        pass

    return PlatformInfo(
        arch=arch,
        kernel_version=kernel_version,
        kunpeng_model=kunpeng_model,
        cpu_model_name=cpu_model,
        socket_count=socket_count,
        cores_per_socket=cores_per_socket,
        threads_per_core=threads_per_core,
        total_cores=online_cpus,
        numa_nodes=numa_nodes,
        l1d_cache_kb=l1d_kb,
        l1i_cache_kb=l1i_kb,
        l2_cache_kb=l2_kb,
        l3_cache_mb=l3_mb,
    )

def is_kunpeng() -> bool:
    return detect_platform().kunpeng_model is not None
