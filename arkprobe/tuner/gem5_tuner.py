"""gem5 simulation tuner for microarchitectural parameter exploration.

This module provides capabilities to:
1. Configure gem5 with different microarchitectural parameters
2. Run simulations with ArkProbe workloads
3. Extract performance metrics from gem5 stats
4. Compare results across configurations

gem5 Parameters Supported:
- L1I/L1D/L2/L3 Cache: size, associativity, latency
- ROB depth, IQ size, LSQ size
- Issue width, commit width, fetch width
- BTB size, RAS size
- Branch predictor type and parameters
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

logger = logging.getLogger(__name__)


class BranchPredictorType(str, Enum):
    """Branch predictor types available in gem5."""
    LOCAL = "LocalBP"
    GLOBAL = "TournamentBP"
    BI_MODAL = "BiModalBP"
    TWO_LEVEL = "TwoLevelBP"


@dataclass
class CacheConfig:
    """Cache configuration parameters."""
    size_kb: int = 64
    assoc: int = 4
    tag_latency: int = 1
    data_latency: int = 1
    response_latency: int = 1
    mshrs: int = 4
    tgts_per_mshr: int = 8

    def to_gem5_size(self) -> str:
        """Convert KB to gem5 size string."""
        if self.size_kb >= 1024:
            return f"{self.size_kb // 1024}MiB"
        return f"{self.size_kb}KiB"


@dataclass
class O3CPUConfig:
    """O3CPU (out-of-order) configuration parameters."""
    # Pipeline widths
    fetch_width: int = 4
    decode_width: int = 4
    rename_width: int = 4
    issue_width: int = 4
    wb_width: int = 4
    commit_width: int = 4

    # Buffer sizes
    fetch_buffer_size: int = 64
    rob_entries: int = 128
    iq_entries: int = 64
    lq_entries: int = 32
    sq_entries: int = 32

    # Branch prediction
    btb_entries: int = 2048
    ras_entries: int = 64
    local_pred_size: int = 2048
    global_pred_size: int = 8192

    # Latencies
    fetch_to_decode_delay: int = 1
    decode_to_rename_delay: int = 1
    rename_to_iew_delay: int = 1
    iew_to_commit_delay: int = 1


@dataclass
class Gem5Config:
    """Complete gem5 simulation configuration.

    Attributes:
        name: Configuration name
        cpu_config: O3CPU parameters
        l1i_cache: L1 instruction cache config
        l1d_cache: L1 data cache config
        l2_cache: L2 cache config (optional)
        l3_cache: L3 cache config (optional)
        cpu_freq: CPU frequency string (e.g., "3GHz")
        mem_size: Memory size string (e.g., "2GB")
        mem_type: Memory controller type
        simulation_time: Max simulation time in seconds
        description: Human-readable description
    """
    name: str = "default"
    cpu_config: O3CPUConfig = field(default_factory=O3CPUConfig)
    l1i_cache: CacheConfig = field(default_factory=lambda: CacheConfig(size_kb=32, assoc=2))
    l1d_cache: CacheConfig = field(default_factory=lambda: CacheConfig(size_kb=32, assoc=2))
    l2_cache: Optional[CacheConfig] = field(default_factory=lambda: CacheConfig(size_kb=256, assoc=8))
    l3_cache: Optional[CacheConfig] = None
    cpu_freq: str = "3GHz"
    mem_size: str = "2GB"
    mem_type: str = "DDR4_2400_8x8"
    simulation_time: float = 0.1  # 100ms simulation time
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "cpu_config": {
                "fetch_width": self.cpu_config.fetch_width,
                "decode_width": self.cpu_config.decode_width,
                "rename_width": self.cpu_config.rename_width,
                "issue_width": self.cpu_config.issue_width,
                "wb_width": self.cpu_config.wb_width,
                "commit_width": self.cpu_config.commit_width,
                "rob_entries": self.cpu_config.rob_entries,
                "iq_entries": self.cpu_config.iq_entries,
                "lq_entries": self.cpu_config.lq_entries,
                "sq_entries": self.cpu_config.sq_entries,
                "btb_entries": self.cpu_config.btb_entries,
                "ras_entries": self.cpu_config.ras_entries,
            },
            "l1i_cache": {
                "size_kb": self.l1i_cache.size_kb,
                "assoc": self.l1i_cache.assoc,
            },
            "l1d_cache": {
                "size_kb": self.l1d_cache.size_kb,
                "assoc": self.l1d_cache.assoc,
            },
            "l2_cache": {
                "size_kb": self.l2_cache.size_kb,
                "assoc": self.l2_cache.assoc,
            } if self.l2_cache else None,
            "cpu_freq": self.cpu_freq,
            "mem_size": self.mem_size,
            "description": self.description,
        }


@dataclass
class Gem5Stats:
    """Parsed gem5 simulation statistics."""
    # Timing
    sim_ticks: int = 0
    sim_seconds: float = 0.0

    # Instructions
    instructions: int = 0
    cycles: int = 0
    ipc: float = 0.0

    # Cache stats
    l1i_accesses: int = 0
    l1i_misses: int = 0
    l1i_mpki: float = 0.0

    l1d_accesses: int = 0
    l1d_misses: int = 0
    l1d_mpki: float = 0.0

    l2_accesses: int = 0
    l2_misses: int = 0
    l2_mpki: float = 0.0

    l3_accesses: int = 0
    l3_misses: int = 0
    l3_mpki: float = 0.0

    # Branch prediction
    branch_predicted: int = 0
    branch_mispredicted: int = 0
    branch_mpki: float = 0.0

    # Pipeline
    rob_occupancy: float = 0.0
    iq_occupancy: float = 0.0
    lsq_occupancy: float = 0.0


class Gem5Tuner:
    """gem5 simulation tuner for microarchitectural exploration.

    Usage:
        tuner = Gem5Tuner(gem5_path="/path/to/gem5")

        # Create a configuration
        config = Gem5Config(
            name="big_rob",
            cpu_config=O3CPUConfig(rob_entries=256),
        )

        # Run simulation
        stats = tuner.simulate(config, binary_path, workload_args)

        # Compare with baseline
        comparison = tuner.compare(baseline_stats, tuned_stats)
    """

    def __init__(
        self,
        gem5_path: Optional[Path] = None,
        gem5_binary: str = "gem5.opt",
        work_dir: Path = Path("./gem5_work"),
    ):
        """Initialize gem5 tuner.

        Args:
            gem5_path: Path to gem5 build directory
            gem5_binary: gem5 binary name (gem5.opt, gem5.fast, etc.)
            work_dir: Working directory for simulation outputs
        """
        self.gem5_path = gem5_path or self._find_gem5()
        self.gem5_binary = gem5_binary
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._check_gem5_available()

    def _find_gem5(self) -> Optional[Path]:
        """Try to find gem5 installation."""
        # Check common locations
        candidates = [
            Path.home() / "gem5" / "build" / "ARM" / "gem5.opt",
            Path("/opt/gem5/build/ARM/gem5.opt"),
            Path("./gem5/build/ARM/gem5.opt"),
        ]

        for path in candidates:
            if path.exists():
                return path.parent.parent.parent

        # Check PATH
        result = shutil.which("gem5.opt")
        if result:
            return Path(result).parent.parent.parent

        return None

    def _check_gem5_available(self):
        """Check if gem5 is available."""
        if self.gem5_path is None:
            logger.warning(
                "gem5 not found. Simulations will fail. "
                "Set gem5_path or install gem5."
            )
            return

        gem5_exe = self.gem5_path / "build" / "ARM" / self.gem5_binary
        if not gem5_exe.exists():
            logger.warning(f"gem5 binary not found: {gem5_exe}")

    def generate_config_script(self, config: Gem5Config) -> Path:
        """Generate gem5 Python configuration script.

        Args:
            config: gem5 configuration

        Returns:
            Path to generated configuration script
        """
        script_path = self.work_dir / f"config_{config.name}.py"
        script_content = self._render_config_template(config)
        script_path.write_text(script_content)
        return script_path

    def _render_config_template(self, config: Gem5Config) -> str:
        """Render gem5 configuration script from template."""
        template = '''#!/usr/bin/env python3
"""gem5 configuration for {name}"""

import argparse
import m5
from m5.objects import *

# Cache Configuration
class L1ICache(BaseCache):
    size = '{l1i_size}'
    assoc = {l1i_assoc}
    tag_latency = {l1i_tag_lat}
    data_latency = {l1i_data_lat}
    response_latency = {l1i_resp_lat}
    mshrs = {l1i_mshrs}
    tgts_per_mshr = {l1i_tgts}
    is_top_level = True

class L1DCache(BaseCache):
    size = '{l1d_size}'
    assoc = {l1d_assoc}
    tag_latency = {l1d_tag_lat}
    data_latency = {l1d_data_lat}
    response_latency = {l1d_resp_lat}
    mshrs = {l1d_mshrs}
    tgts_per_mshr = {l1d_tgts}
    write_buffers = 16
    is_top_level = True

{l2_cache_class}

# CPU Configuration
class O3_ARM_CPU(DerivO3CPU):
    # Pipeline widths
    fetchWidth = {fetch_width}
    decodeWidth = {decode_width}
    renameWidth = {rename_width}
    issueWidth = {issue_width}
    wbWidth = {wb_width}
    commitWidth = {commit_width}

    # Buffer sizes
    fetchBufferSize = {fetch_buffer_size}
    numROBEntries = {rob_entries}
    numIQEntries = {iq_entries}
    LQEntries = {lq_entries}
    SQEntries = {sq_entries}

    # Branch prediction
    BTBEntries = {btb_entries}
    RASSize = {ras_entries}

    # Latencies
    fetchToDecodeDelay = {fetch_to_decode}
    decodeToRenameDelay = {decode_to_rename}
    renameToIEWDelay = {rename_to_iew}
    IEWToCommitDelay = {iew_to_commit}

def create_system(binary, args):
    """Create the simulated system."""
    system = System(cpu_clk_domain=SrcClockDomain(clock='{cpu_freq}',
                                                   voltage_domain=VoltageDomain()))

    # Memory
    system.mem_mode = 'timing'
    system.mem_ranges = [AddrRange('{mem_size}')]

    # CPU
    system.cpu = O3_ARM_CPU(clk_domain=system.cpu_clk_domain)

    # Caches
    system.cpu.icache = L1ICache()
    system.cpu.dcache = L1DCache()

    {l2_cache_inst}

    # Memory bus
    system.membus = SystemXBar()

    # Connect caches to memory
    system.cpu.icache.cpu_side = system.cpu.icache_port
    system.cpu.dcache.cpu_side = system.cpu.dcache_port

    {l2_connections}

    system.system_port = system.membus.cpu_side_ports

    # Memory controller
    system.mem_ctrl = {mem_type}()
    system.mem_ctrl.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports

    # Workload
    system.workload = SEWorkload.init_compatible(binary)

    # Process
    process = Process()
    process.cmd = [binary] + args
    system.cpu.workload[0] = process
    system.cpu.createThreads()

    return system

def main():
    parser = argparse.ArgumentParser(description='gem5 config: {name}')
    parser.add_argument('binary', help='Binary to execute')
    parser.add_argument('--args', nargs='*', default=[], help='Binary arguments')
    parser.add_argument('--max-time', type=float, default={sim_time},
                        help='Max simulation time in seconds')
    args = parser.parse_args()

    root = Root(full_system=False)
    root.system = create_system(args.binary, args.args)

    # Run simulation
    m5.instantiate()

    # Simulate for specified time
    exit_event = m5.simulate(root.system.cpu_clk_domain.clock.period * args.max_time * 1e12)

    print(f"Simulation ended at tick {{m5.curTick()}}")
    print(f"Exit reason: {{exit_event.getCause()}}")

    # Print stats
    stats_file = open('stats.txt', 'w')
    for stat in root.system.cpu.stats:
        stats_file.write(f"{{{{stat.name}}}}: {{{{stat.value}}}}\\n")
    stats_file.close()

if __name__ == '__main__':
    main()
'''
        # Format cache configurations
        l2_cache_class = ""
        l2_cache_inst = ""
        l2_connections = """system.cpu.icache.mem_side = system.membus.cpu_side_ports
    system.cpu.dcache.mem_side = system.membus.cpu_side_ports"""

        if config.l2_cache:
            l2_cache_class = f"""class L2Cache(BaseCache):
    size = '{config.l2_cache.to_gem5_size()}'
    assoc = {config.l2_cache.assoc}
    tag_latency = {config.l2_cache.tag_latency}
    data_latency = {config.l2_cache.data_latency}
    response_latency = {config.l2_cache.response_latency}
    mshrs = {config.l2_cache.mshrs}
    tgts_per_mshr = {config.l2_cache.tgts_per_mshr}"""

            l2_cache_inst = "system.l2cache = L2Cache()"
            l2_connections = """system.l2bus = L2XBar()

    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l2cache.mem_side = system.membus.cpu_side_ports"""

        return template.format(
            name=config.name,
            l1i_size=config.l1i_cache.to_gem5_size(),
            l1i_assoc=config.l1i_cache.assoc,
            l1i_tag_lat=config.l1i_cache.tag_latency,
            l1i_data_lat=config.l1i_cache.data_latency,
            l1i_resp_lat=config.l1i_cache.response_latency,
            l1i_mshrs=config.l1i_cache.mshrs,
            l1i_tgts=config.l1i_cache.tgts_per_mshr,
            l1d_size=config.l1d_cache.to_gem5_size(),
            l1d_assoc=config.l1d_cache.assoc,
            l1d_tag_lat=config.l1d_cache.tag_latency,
            l1d_data_lat=config.l1d_cache.data_latency,
            l1d_resp_lat=config.l1d_cache.response_latency,
            l1d_mshrs=config.l1d_cache.mshrs,
            l1d_tgts=config.l1d_cache.tgts_per_mshr,
            l2_cache_class=l2_cache_class,
            l2_cache_inst=l2_cache_inst,
            l2_connections=l2_connections,
            fetch_width=config.cpu_config.fetch_width,
            decode_width=config.cpu_config.decode_width,
            rename_width=config.cpu_config.rename_width,
            issue_width=config.cpu_config.issue_width,
            wb_width=config.cpu_config.wb_width,
            commit_width=config.cpu_config.commit_width,
            fetch_buffer_size=config.cpu_config.fetch_buffer_size,
            rob_entries=config.cpu_config.rob_entries,
            iq_entries=config.cpu_config.iq_entries,
            lq_entries=config.cpu_config.lq_entries,
            sq_entries=config.cpu_config.sq_entries,
            btb_entries=config.cpu_config.btb_entries,
            ras_entries=config.cpu_config.ras_entries,
            fetch_to_decode=config.cpu_config.fetch_to_decode_delay,
            decode_to_rename=config.cpu_config.decode_to_rename_delay,
            rename_to_iew=config.cpu_config.rename_to_iew_delay,
            iew_to_commit=config.cpu_config.iew_to_commit_delay,
            cpu_freq=config.cpu_freq,
            mem_size=config.mem_size,
            mem_type=config.mem_type,
            sim_time=config.simulation_time,
        )

    def simulate(
        self,
        config: Gem5Config,
        binary_path: Path,
        workload_args: list[str] = None,
        timeout: int = 3600,
    ) -> Gem5Stats:
        """Run gem5 simulation.

        Args:
            config: gem5 configuration
            binary_path: Path to binary to simulate
            workload_args: Arguments to pass to binary
            timeout: Maximum wall-clock time in seconds

        Returns:
            Gem5Stats with simulation results
        """
        if self.gem5_path is None:
            raise RuntimeError("gem5 not available")

        workload_args = workload_args or []

        # Generate config script
        config_script = self.generate_config_script(config)

        # Create output directory
        output_dir = self.work_dir / f"sim_{config.name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build gem5 command
        gem5_exe = self.gem5_path / "build" / "ARM" / self.gem5_binary
        cmd = [
            str(gem5_exe),
            "-d", str(output_dir),
            str(config_script),
            str(binary_path),
            "--args", *workload_args,
        ]

        logger.info(f"Running gem5: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=output_dir,
            )

            if result.returncode != 0:
                logger.error(f"gem5 failed: {result.stderr}")
                return Gem5Stats()

            # Parse stats
            stats_file = output_dir / "stats.txt"
            if stats_file.exists():
                return self.parse_stats(stats_file)
            else:
                logger.warning("No stats.txt found")
                return Gem5Stats()

        except subprocess.TimeoutExpired:
            logger.error(f"gem5 simulation timed out after {timeout}s")
            return Gem5Stats()
        except Exception as e:
            logger.error(f"gem5 simulation error: {e}")
            return Gem5Stats()

    def parse_stats(self, stats_file: Path) -> Gem5Stats:
        """Parse gem5 statistics file.

        Args:
            stats_file: Path to stats.txt

        Returns:
            Gem5Stats with parsed values
        """
        stats = Gem5Stats()
        content = stats_file.read_text()

        # Parse key-value pairs
        values = {}
        for line in content.split('\n'):
            if ':' in line:
                # Split only on first colon to handle :: in stat names
                idx = line.find(':')
                # But we need to handle ::total style names
                # Check if this is ::total or similar
                if idx > 0 and line[idx:idx+2] == '::':
                    # Find the next single colon after ::
                    idx = line.find(':', idx+2)
                if idx > 0:
                    name = line[:idx].strip()
                    value_str = line[idx+1:].strip().split()[0] if line[idx+1:].strip() else ''
                    try:
                        value = float(value_str)
                        values[name] = value
                    except (ValueError, IndexError):
                        continue

        # Extract key statistics
        # Timing
        stats.sim_ticks = int(values.get('simTicks', 0))
        stats.sim_seconds = values.get('simSeconds', 0.0)

        # Instructions and cycles
        stats.instructions = int(values.get('system.cpu.committedInsts', 0))
        stats.cycles = int(values.get('system.cpu.numCycles', 0))

        if stats.cycles > 0:
            stats.ipc = stats.instructions / stats.cycles

        # L1I cache
        stats.l1i_accesses = int(values.get('system.cpu.icache.demandAccesses::total', 0))
        stats.l1i_misses = int(values.get('system.cpu.icache.demandMisses::total', 0))
        if stats.instructions > 0:
            stats.l1i_mpki = stats.l1i_misses / (stats.instructions / 1000)

        # L1D cache
        stats.l1d_accesses = int(values.get('system.cpu.dcache.demandAccesses::total', 0))
        stats.l1d_misses = int(values.get('system.cpu.dcache.demandMisses::total', 0))
        if stats.instructions > 0:
            stats.l1d_mpki = stats.l1d_misses / (stats.instructions / 1000)

        # L2 cache
        stats.l2_accesses = int(values.get('system.l2cache.demandAccesses::total', 0))
        stats.l2_misses = int(values.get('system.l2cache.demandMisses::total', 0))
        if stats.instructions > 0:
            stats.l2_mpki = stats.l2_misses / (stats.instructions / 1000)

        # Branch prediction
        stats.branch_predicted = int(values.get('system.cpu.branchPred.lookups', 0))
        stats.branch_mispredicted = int(values.get('system.cpu.branchPred.mispredicted', 0))
        if stats.instructions > 0:
            stats.branch_mpki = stats.branch_mispredicted / (stats.instructions / 1000)

        return stats

    def stats_to_feature_dict(self, stats: Gem5Stats) -> dict:
        """Convert gem5 stats to feature dictionary.

        This allows comparison with real hardware feature vectors.
        """
        return {
            "ipc": stats.ipc,
            "instructions": stats.instructions,
            "cycles": stats.cycles,
            "l1i_mpki": stats.l1i_mpki,
            "l1d_mpki": stats.l1d_mpki,
            "l2_mpki": stats.l2_mpki,
            "l3_mpki": stats.l3_mpki,
            "branch_mpki": stats.branch_mpki,
            "sim_seconds": stats.sim_seconds,
        }


# Predefined gem5 configurations
GEM5_PRESETS = {
    "default": Gem5Config(
        name="default",
        description="Default gem5 configuration (similar to ARM Cortex-A72)",
    ),
    "small_cache": Gem5Config(
        name="small_cache",
        l1i_cache=CacheConfig(size_kb=16, assoc=2),
        l1d_cache=CacheConfig(size_kb=16, assoc=2),
        l2_cache=CacheConfig(size_kb=128, assoc=4),
        description="Small caches for area-constrained design",
    ),
    "large_cache": Gem5Config(
        name="large_cache",
        l1i_cache=CacheConfig(size_kb=64, assoc=4),
        l1d_cache=CacheConfig(size_kb=64, assoc=4),
        l2_cache=CacheConfig(size_kb=512, assoc=8),
        description="Large caches for performance",
    ),
    "wide_issue": Gem5Config(
        name="wide_issue",
        cpu_config=O3CPUConfig(
            fetch_width=8,
            decode_width=8,
            rename_width=8,
            issue_width=8,
            wb_width=8,
            commit_width=8,
        ),
        description="8-wide issue machine",
    ),
    "deep_rob": Gem5Config(
        name="deep_rob",
        cpu_config=O3CPUConfig(rob_entries=256, iq_entries=128),
        description="Deep ROB for memory latency tolerance",
    ),
    "big_btb": Gem5Config(
        name="big_btb",
        cpu_config=O3CPUConfig(btb_entries=8192, ras_entries=256),
        description="Large BTB for branch-heavy code",
    ),
    "kunpeng_like": Gem5Config(
        name="kunpeng_like",
        cpu_config=O3CPUConfig(
            fetch_width=4,
            decode_width=4,
            rename_width=4,
            issue_width=4,
            wb_width=4,
            commit_width=4,
            rob_entries=128,
            iq_entries=64,
            btb_entries=2048,
        ),
        l1i_cache=CacheConfig(size_kb=64, assoc=4),
        l1d_cache=CacheConfig(size_kb=64, assoc=4),
        l2_cache=CacheConfig(size_kb=512, assoc=8),
        description="Configuration similar to Kunpeng 920",
    ),
}
