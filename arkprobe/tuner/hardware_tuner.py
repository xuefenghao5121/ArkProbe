"""Hardware parameter tuner for real system configuration.

Supports tuning:
- CPU frequency (governor + fixed frequency)
- SMT (Hyperthreading) on/off
- C-state limits
- NUMA policy
- CPU affinity
- Transparent Huge Pages
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CPUGovernor(str, Enum):
    """CPU frequency governor options."""
    PERFORMANCE = "performance"
    POWERSAVE = "powersave"
    USERSPACE = "userspace"
    ONDEMAND = "ondemand"
    CONSERVATIVE = "conservative"
    SCHEDUTIL = "schedutil"


class CStateLimit(int, Enum):
    """C-state depth limits."""
    C0_ONLY = 0      # No idle states
    C1_MAX = 1       # Halt only
    C2_MAX = 2       # Allow deeper sleep
    UNLIMITED = -1   # No limit


class NUMAPolicy(str, Enum):
    """NUMA memory allocation policies."""
    DEFAULT = "default"
    BIND = "bind"
    INTERLEAVE = "interleave"
    PREFERRED = "preferred"


class THPSetting(str, Enum):
    """Transparent Huge Page settings."""
    ALWAYS = "always"
    MADVISE = "madvise"
    NEVER = "never"


@dataclass
class TuningConfig:
    """Configuration for hardware tuning experiment.

    Attributes:
        name: Human-readable config name
        cpu_governor: CPU frequency governor
        cpu_frequency_mhz: Fixed frequency (only for userspace governor)
        smt_enabled: SMT (hyperthreading) on/off
        cstate_limit: Maximum C-state depth
        numa_policy: NUMA memory policy
        numa_nodes: Specific NUMA nodes (for bind/interleave)
        cpu_affinity: CPU cores to bind (list of CPU IDs)
        thp_setting: Transparent huge page setting
        description: Human-readable description
    """
    name: str = "default"
    cpu_governor: CPUGovernor = CPUGovernor.PERFORMANCE
    cpu_frequency_mhz: Optional[int] = None
    smt_enabled: bool = True
    cstate_limit: CStateLimit = CStateLimit.UNLIMITED
    numa_policy: NUMAPolicy = NUMAPolicy.DEFAULT
    numa_nodes: Optional[list[int]] = None
    cpu_affinity: Optional[list[int]] = None
    thp_setting: THPSetting = THPSetting.MADVISE
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "cpu_governor": self.cpu_governor.value,
            "cpu_frequency_mhz": self.cpu_frequency_mhz,
            "smt_enabled": self.smt_enabled,
            "cstate_limit": self.cstate_limit.value,
            "numa_policy": self.numa_policy.value,
            "numa_nodes": self.numa_nodes,
            "cpu_affinity": self.cpu_affinity,
            "thp_setting": self.thp_setting.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TuningConfig":
        """Create from dictionary."""
        return cls(
            name=d.get("name", "default"),
            cpu_governor=CPUGovernor(d.get("cpu_governor", "performance")),
            cpu_frequency_mhz=d.get("cpu_frequency_mhz"),
            smt_enabled=d.get("smt_enabled", True),
            cstate_limit=CStateLimit(d.get("cstate_limit", -1)),
            numa_policy=NUMAPolicy(d.get("numa_policy", "default")),
            numa_nodes=d.get("numa_nodes"),
            cpu_affinity=d.get("cpu_affinity"),
            thp_setting=THPSetting(d.get("thp_setting", "madvise")),
            description=d.get("description", ""),
        )

    @classmethod
    def load_yaml(cls, path: Path) -> "TuningConfig":
        """Load from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("tuning", {}))


@dataclass
class SystemState:
    """Snapshot of current system configuration."""
    cpu_governor: str
    cpu_frequency_mhz: dict[int, int]  # CPU ID -> frequency
    smt_enabled: bool
    cstate_states: dict[str, bool]  # C-state name -> disabled
    thp_enabled: str
    numa_nodes: list[int]
    online_cpus: list[int]


@dataclass
class TuningResult:
    """Result of a tuning experiment.

    Attributes:
        config: Applied tuning configuration
        success: Whether tuning was successful
        errors: List of error messages
        original_state: System state before tuning
        applied_state: System state after tuning
    """
    config: TuningConfig
    success: bool = True
    errors: list[str] = field(default_factory=list)
    original_state: Optional[SystemState] = None
    applied_state: Optional[SystemState] = None


class HardwareTuner:
    """Tune real hardware parameters on Kunpeng systems.

    Usage:
        tuner = HardwareTuner()

        # Create a tuning config
        config = TuningConfig(
            name="high_perf",
            cpu_governor=CPUGovernor.PERFORMANCE,
            smt_enabled=False,
            cstate_limit=CStateLimit.C1_MAX,
        )

        # Apply tuning
        result = tuner.apply(config)

        # Run workload...

        # Restore original state
        tuner.restore()
    """

    CPU_BASE_PATH = Path("/sys/devices/system/cpu")
    THP_PATH = Path("/sys/kernel/mm/transparent_hugepage/enabled")

    def __init__(self, dry_run: bool = False):
        """Initialize tuner.

        Args:
            dry_run: If True, don't actually change system settings
        """
        self.dry_run = dry_run
        self._saved_state: Optional[SystemState] = None
        self._check_permissions()

    def _check_permissions(self):
        """Check if we have necessary permissions."""
        if os.geteuid() != 0 and not self.dry_run:
            logger.warning(
                "Not running as root. Some tuning operations may fail. "
                "Use --dry-run to preview changes."
            )

    def get_current_state(self) -> SystemState:
        """Capture current system configuration."""
        # CPU governor
        governor = self._get_cpu_governor()

        # CPU frequencies
        frequencies = self._get_cpu_frequencies()

        # SMT status
        smt_enabled = self._get_smt_status()

        # C-state status
        cstate_states = self._get_cstate_status()

        # THP status
        thp_enabled = self._get_thp_status()

        # NUMA nodes
        numa_nodes = self._get_numa_nodes()

        # Online CPUs
        online_cpus = self._get_online_cpus()

        return SystemState(
            cpu_governor=governor,
            cpu_frequency_mhz=frequencies,
            smt_enabled=smt_enabled,
            cstate_states=cstate_states,
            thp_enabled=thp_enabled,
            numa_nodes=numa_nodes,
            online_cpus=online_cpus,
        )

    def _get_cpu_governor(self) -> str:
        """Get current CPU frequency governor."""
        path = self.CPU_BASE_PATH / "cpu0" / "cpufreq" / "scaling_governor"
        if path.exists():
            return path.read_text().strip()
        return "unknown"

    def _get_cpu_frequencies(self) -> dict[int, int]:
        """Get current CPU frequencies in MHz."""
        frequencies = {}
        for cpu_path in self.CPU_BASE_PATH.glob("cpu[0-9]*"):
            cpu_id = int(cpu_path.name[3:])
            freq_path = cpu_path / "cpufreq" / "scaling_cur_freq"
            if freq_path.exists():
                freq_khz = int(freq_path.read_text().strip())
                frequencies[cpu_id] = freq_khz // 1000
        return frequencies

    def _get_smt_status(self) -> bool:
        """Get SMT (hyperthreading) status."""
        path = self.CPU_BASE_PATH / "smt" / "active"
        if path.exists():
            return path.read_text().strip() == "1"
        # Fallback: check if thread siblings exist
        return len(self._get_online_cpus()) > self._get_core_count()

    def _get_core_count(self) -> int:
        """Get number of physical cores."""
        path = Path("/proc/cpuinfo")
        if path.exists():
            content = path.read_text()
            # Count unique core IDs
            core_ids = set(re.findall(r"core id\s*:\s*(\d+)", content))
            return len(core_ids) if core_ids else 1
        return 1

    def _get_cstate_status(self) -> dict[str, bool]:
        """Get C-state disable status."""
        cstate_states = {}
        for cpu_path in self.CPU_BASE_PATH.glob("cpu[0-9]*"):
            for cstate_path in (cpu_path / "cpuidle").glob("state*"):
                name_path = cstate_path / "name"
                disable_path = cstate_path / "disable"
                if name_path.exists() and disable_path.exists():
                    name = name_path.read_text().strip()
                    disabled = disable_path.read_text().strip() == "1"
                    cstate_states[name] = disabled
            break  # Only check first CPU
        return cstate_states

    def _get_thp_status(self) -> str:
        """Get Transparent Huge Page status."""
        if self.THP_PATH.exists():
            content = self.THP_PATH.read_text()
            # Format: "always [madvise] never"
            match = re.search(r"\[(\w+)\]", content)
            if match:
                return match.group(1)
        return "unknown"

    def _get_numa_nodes(self) -> list[int]:
        """Get available NUMA nodes."""
        nodes = []
        numa_path = Path("/sys/devices/system/node")
        if numa_path.exists():
            for node_path in numa_path.glob("node[0-9]*"):
                node_id = int(node_path.name[4:])
                nodes.append(node_id)
        return sorted(nodes) if nodes else [0]

    def _get_online_cpus(self) -> list[int]:
        """Get list of online CPUs."""
        cpus = []
        online_path = self.CPU_BASE_PATH / "online"
        if online_path.exists():
            content = online_path.read_text().strip()
            # Parse format like "0-63" or "0-31,64-95"
            for part in content.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    cpus.extend(range(start, end + 1))
                else:
                    cpus.append(int(part))
        return sorted(cpus)

    def apply(self, config: TuningConfig) -> TuningResult:
        """Apply tuning configuration.

        Args:
            config: Tuning configuration to apply

        Returns:
            TuningResult with success status and any errors
        """
        errors = []

        # Save current state for restoration
        original_state = self.get_current_state()
        self._saved_state = original_state

        logger.info(f"Applying tuning config: {config.name}")

        # Apply each setting
        if config.cpu_governor:
            err = self._set_cpu_governor(config.cpu_governor)
            if err:
                errors.append(err)

        if config.cpu_frequency_mhz:
            err = self._set_cpu_frequency(config.cpu_frequency_mhz)
            if err:
                errors.append(err)

        if not config.smt_enabled:
            err = self._disable_smt()
            if err:
                errors.append(err)

        if config.cstate_limit != CStateLimit.UNLIMITED:
            err = self._limit_cstates(config.cstate_limit)
            if err:
                errors.append(err)

        if config.thp_setting:
            err = self._set_thp(config.thp_setting)
            if err:
                errors.append(err)

        # Capture applied state
        applied_state = self.get_current_state() if not self.dry_run else None

        return TuningResult(
            config=config,
            success=len(errors) == 0,
            errors=errors,
            original_state=original_state,
            applied_state=applied_state,
        )

    def _set_cpu_governor(self, governor: CPUGovernor) -> Optional[str]:
        """Set CPU frequency governor."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would set governor to {governor.value}")
            return None

        try:
            # Use cpupower if available
            if shutil.which("cpupower"):
                subprocess.run(
                    ["cpupower", "frequency-set", "-g", governor.value],
                    check=True, capture_output=True,
                )
                logger.info(f"Set CPU governor to {governor.value}")
                return None
            else:
                # Direct sysfs write
                for cpu_path in self.CPU_BASE_PATH.glob("cpu[0-9]*"):
                    gov_path = cpu_path / "cpufreq" / "scaling_governor"
                    if gov_path.exists():
                        gov_path.write_text(governor.value)
                logger.info(f"Set CPU governor to {governor.value} (sysfs)")
                return None
        except Exception as e:
            return f"Failed to set governor: {e}"

    def _set_cpu_frequency(self, freq_mhz: int) -> Optional[str]:
        """Set fixed CPU frequency (requires userspace governor)."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would set frequency to {freq_mhz} MHz")
            return None

        try:
            freq_khz = freq_mhz * 1000
            if shutil.which("cpupower"):
                subprocess.run(
                    ["cpupower", "frequency-set", "-f", str(freq_mhz)],
                    check=True, capture_output=True,
                )
                logger.info(f"Set CPU frequency to {freq_mhz} MHz")
                return None
            else:
                for cpu_path in self.CPU_BASE_PATH.glob("cpu[0-9]*"):
                    freq_path = cpu_path / "cpufreq" / "scaling_setspeed"
                    if freq_path.exists():
                        freq_path.write_text(str(freq_khz))
                logger.info(f"Set CPU frequency to {freq_mhz} MHz (sysfs)")
                return None
        except Exception as e:
            return f"Failed to set frequency: {e}"

    def _disable_smt(self) -> Optional[str]:
        """Disable SMT (hyperthreading)."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would disable SMT")
            return None

        try:
            smt_control = self.CPU_BASE_PATH / "smt" / "control"
            if smt_control.exists():
                # Check if SMT is supported
                smt_active = self.CPU_BASE_PATH / "smt" / "active"
                if smt_active.exists():
                    active = smt_active.read_text().strip()
                    if active == "0":
                        logger.info("SMT not active, skipping disable")
                        return None
                smt_control.write_text("off")
                logger.info("Disabled SMT")
                return None
            else:
                logger.info("SMT control not available in kernel")
                return None  # Not an error, just not supported
        except Exception as e:
            return f"Failed to disable SMT: {e}"

    def _enable_smt(self) -> Optional[str]:
        """Enable SMT (hyperthreading)."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would enable SMT")
            return None

        try:
            smt_control = self.CPU_BASE_PATH / "smt" / "control"
            if smt_control.exists():
                smt_control.write_text("on")
                logger.info("Enabled SMT")
                return None
            else:
                logger.info("SMT control not available in kernel")
                return None  # Not an error, just not supported
        except Exception as e:
            return f"Failed to enable SMT: {e}"

    def _limit_cstates(self, limit: CStateLimit) -> Optional[str]:
        """Limit C-state depth."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would limit C-states to C{limit.value}")
            return None

        try:
            for cpu_path in self.CPU_BASE_PATH.glob("cpu[0-9]*"):
                for cstate_path in (cpu_path / "cpuidle").glob("state*"):
                    # Read state index/name
                    name_path = cstate_path / "name"
                    if not name_path.exists():
                        continue

                    name = name_path.read_text().strip()
                    # C-state names like "C0", "C1", "C2", etc.
                    match = re.match(r"C(\d+)", name)
                    if match:
                        state_depth = int(match.group(1))
                        disable_path = cstate_path / "disable"
                        # Disable if deeper than limit
                        should_disable = state_depth > limit.value
                        if disable_path.exists():
                            disable_path.write_text("1" if should_disable else "0")

            logger.info(f"Limited C-states to C{limit.value}")
            return None
        except Exception as e:
            return f"Failed to limit C-states: {e}"

    def _set_thp(self, setting: THPSetting) -> Optional[str]:
        """Set Transparent Huge Page setting."""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would set THP to {setting.value}")
            return None

        try:
            if self.THP_PATH.exists():
                self.THP_PATH.write_text(setting.value)
                logger.info(f"Set THP to {setting.value}")
                return None
            else:
                return "THP not available in kernel"
        except Exception as e:
            return f"Failed to set THP: {e}"

    def restore(self) -> bool:
        """Restore original system state.

        Returns:
            True if restoration was successful
        """
        if self._saved_state is None:
            logger.warning("No saved state to restore")
            return True

        if self.dry_run:
            logger.info("[DRY-RUN] Would restore original state")
            return True

        logger.info("Restoring original system state...")
        errors = []

        # Restore governor
        err = self._set_cpu_governor(CPUGovernor(self._saved_state.cpu_governor))
        if err:
            errors.append(err)

        # Restore SMT
        if self._saved_state.smt_enabled:
            err = self._enable_smt()
        else:
            err = self._disable_smt()
        if err:
            errors.append(err)

        # Restore THP
        if self._saved_state.thp_enabled != "unknown":
            err = self._set_thp(THPSetting(self._saved_state.thp_enabled))
            if err:
                errors.append(err)

        # Restore C-states (enable all)
        for cpu_path in self.CPU_BASE_PATH.glob("cpu[0-9]*"):
            for cstate_path in (cpu_path / "cpuidle").glob("state*"):
                disable_path = cstate_path / "disable"
                if disable_path.exists():
                    disable_path.write_text("0")

        if errors:
            logger.error(f"Errors during restoration: {errors}")
            return False

        logger.info("Original state restored")
        self._saved_state = None
        return True

    def get_numactl_cmd(self, config: TuningConfig) -> list[str]:
        """Generate numactl command prefix for the config.

        Args:
            config: Tuning configuration

        Returns:
            List of command arguments for numactl
        """
        if config.numa_policy == NUMAPolicy.DEFAULT:
            return []

        cmd = ["numactl"]

        if config.numa_policy == NUMAPolicy.BIND and config.numa_nodes:
            cmd.extend(["--cpunodebind=" + ",".join(map(str, config.numa_nodes))])
            cmd.extend(["--membind=" + ",".join(map(str, config.numa_nodes))])
        elif config.numa_policy == NUMAPolicy.INTERLEAVE and config.numa_nodes:
            cmd.extend(["--interleave=" + ",".join(map(str, config.numa_nodes))])
        elif config.numa_policy == NUMAPolicy.PREFERRED and config.numa_nodes:
            cmd.extend(["--preferred=" + str(config.numa_nodes[0])])

        return cmd

    def get_taskset_cmd(self, config: TuningConfig) -> list[str]:
        """Generate taskset command prefix for CPU affinity.

        Args:
            config: Tuning configuration

        Returns:
            List of command arguments for taskset
        """
        if not config.cpu_affinity:
            return []

        # Convert CPU list to hex mask
        mask = 0
        for cpu in config.cpu_affinity:
            mask |= (1 << cpu)

        return ["taskset", hex(mask)]

    def wrap_command(self, config: TuningConfig, cmd: list[str]) -> list[str]:
        """Wrap a command with numactl and taskset as needed.

        Args:
            config: Tuning configuration
            cmd: Original command

        Returns:
            Wrapped command with affinity/NUMA settings
        """
        wrapped = []

        # Add numactl prefix
        numactl = self.get_numactl_cmd(config)
        wrapped.extend(numactl)

        # Add taskset prefix
        taskset = self.get_taskset_cmd(config)
        wrapped.extend(taskset)

        # Add original command
        wrapped.extend(cmd)

        return wrapped


# Predefined tuning configurations
TUNING_PRESETS = {
    "default": TuningConfig(
        name="default",
        description="Default system configuration (no changes)",
    ),
    "performance": TuningConfig(
        name="performance",
        cpu_governor=CPUGovernor.PERFORMANCE,
        smt_enabled=True,
        cstate_limit=CStateLimit.C1_MAX,
        thp_setting=THPSetting.ALWAYS,
        description="Maximum performance: performance governor, SMT on, limited C-states",
    ),
    "performance_no_smt": TuningConfig(
        name="performance_no_smt",
        cpu_governor=CPUGovernor.PERFORMANCE,
        smt_enabled=False,
        cstate_limit=CStateLimit.C1_MAX,
        thp_setting=THPSetting.ALWAYS,
        description="Performance without SMT: for lock-heavy workloads",
    ),
    "latency": TuningConfig(
        name="latency",
        cpu_governor=CPUGovernor.PERFORMANCE,
        smt_enabled=False,
        cstate_limit=CStateLimit.C0_ONLY,
        thp_setting=THPSetting.MADVISE,
        description="Minimum latency: no C-states, no SMT",
    ),
    "power": TuningConfig(
        name="power",
        cpu_governor=CPUGovernor.POWERSAVE,
        smt_enabled=True,
        cstate_limit=CStateLimit.UNLIMITED,
        thp_setting=THPSetting.MADVISE,
        description="Power saving: powersave governor, deep C-states",
    ),
    "database": TuningConfig(
        name="database",
        cpu_governor=CPUGovernor.PERFORMANCE,
        smt_enabled=False,
        cstate_limit=CStateLimit.C1_MAX,
        thp_setting=THPSetting.ALWAYS,
        description="Database optimized: no SMT to reduce lock contention",
    ),
    "compute": TuningConfig(
        name="compute",
        cpu_governor=CPUGovernor.PERFORMANCE,
        smt_enabled=True,
        cstate_limit=CStateLimit.C1_MAX,
        thp_setting=THPSetting.ALWAYS,
        description="Compute intensive: full parallelism",
    ),
    "memory": TuningConfig(
        name="memory",
        cpu_governor=CPUGovernor.PERFORMANCE,
        smt_enabled=True,
        cstate_limit=CStateLimit.C1_MAX,
        thp_setting=THPSetting.ALWAYS,
        description="Memory intensive: THP for large pages",
    ),
}
