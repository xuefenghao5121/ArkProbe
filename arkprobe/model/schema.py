"""Unified workload feature vector schema.

This is the central data contract of the framework. Every workload scenario,
regardless of type, is profiled into this same structure so that a chip architect
can directly compare workloads along identical axes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import AccessPattern, IPCMechanism, ScenarioType


# ---------------------------------------------------------------------------
# TopDown methodology models
# ---------------------------------------------------------------------------

class TopDownL1(BaseModel):
    """ARM TopDown Level 1 breakdown (fractions summing to ~1.0)."""
    frontend_bound: float = Field(..., ge=0.0, le=1.0)
    backend_bound: float = Field(..., ge=0.0, le=1.0)
    retiring: float = Field(..., ge=0.0, le=1.0)
    bad_speculation: float = Field(..., ge=0.0, le=1.0)


class TopDownL2(BaseModel):
    """ARM TopDown Level 2 sub-categories."""
    # Frontend
    fetch_latency: Optional[float] = None
    fetch_bandwidth: Optional[float] = None
    # Backend
    memory_bound: Optional[float] = None
    core_bound: Optional[float] = None
    # Bad Speculation
    branch_mispredicts: Optional[float] = None
    machine_clears: Optional[float] = None


# ---------------------------------------------------------------------------
# Feature dimension models
# ---------------------------------------------------------------------------

class InstructionMix(BaseModel):
    """Instruction type breakdown (fractions summing to ~1.0)."""
    integer_ratio: float = Field(..., description="Integer ALU fraction")
    fp_ratio: float = Field(..., description="Floating-point fraction")
    vector_ratio: float = Field(..., description="NEON/SVE vector fraction")
    branch_ratio: float = Field(..., description="Branch fraction")
    load_ratio: float = Field(..., description="Load fraction")
    store_ratio: float = Field(..., description="Store fraction")
    other_ratio: float = Field(0.0, description="Other (barriers, system, etc.)")


class ComputeCharacteristics(BaseModel):
    """Instruction-level compute profile."""
    ipc: float = Field(..., description="Instructions Per Cycle")
    cpi: float = Field(..., description="Cycles Per Instruction")
    instruction_mix: InstructionMix
    simd_utilization: float = Field(0.0, ge=0.0, le=1.0,
                                    description="Fraction of NEON/SVE instructions")
    topdown_l1: TopDownL1
    topdown_l2: Optional[TopDownL2] = None


class CacheHierarchy(BaseModel):
    """Cache performance across all levels."""
    l1i_mpki: float = Field(..., description="L1I misses per kilo-instructions")
    l1d_mpki: float = Field(..., description="L1D misses per kilo-instructions")
    l2_mpki: float = Field(..., description="L2 MPKI")
    l3_mpki: float = Field(..., description="L3/LLC MPKI")
    l1d_miss_rate: float = Field(..., description="L1D misses / accesses")
    l2_miss_rate: float = Field(..., description="L2 misses / accesses")
    l3_miss_rate: float = Field(..., description="L3 misses / accesses")
    spatial_locality_score: Optional[float] = Field(None, ge=0.0, le=1.0,
                                                    description="0=random, 1=perfect spatial")
    temporal_locality_score: Optional[float] = Field(None, ge=0.0, le=1.0,
                                                     description="0=no reuse, 1=high reuse")
    working_set_size_bytes: Optional[int] = Field(None, description="Estimated working set")


class BranchBehavior(BaseModel):
    """Branch prediction characteristics."""
    branch_mpki: float = Field(..., description="Mispredicts per kilo-instructions")
    branch_mispredict_rate: float = Field(..., ge=0.0, le=1.0)
    indirect_branch_ratio: Optional[float] = Field(None,
                                                    description="Indirect branch fraction")
    branch_density: Optional[float] = Field(None, description="Branches per instruction")


class MemorySubsystem(BaseModel):
    """Memory bandwidth and latency profile."""
    bandwidth_read_gbps: float = Field(..., description="Read bandwidth GB/s")
    bandwidth_write_gbps: float = Field(..., description="Write bandwidth GB/s")
    bandwidth_utilization: float = Field(..., ge=0.0, le=1.0,
                                        description="Fraction of peak BW used")
    avg_latency_ns: Optional[float] = Field(None, description="Average access latency ns")
    p99_latency_ns: Optional[float] = Field(None, description="P99 access latency ns")
    access_pattern: Optional[AccessPattern] = None
    numa_local_ratio: Optional[float] = Field(None, ge=0.0, le=1.0,
                                              description="Local NUMA access fraction")
    tlb_mpki: Optional[float] = Field(None, description="TLB MPKI (dTLB + iTLB)")


class IOCharacteristics(BaseModel):
    """Disk / storage I/O profile."""
    iops_read: float = Field(0.0)
    iops_write: float = Field(0.0)
    throughput_read_mbps: float = Field(0.0)
    throughput_write_mbps: float = Field(0.0)
    avg_latency_us: Optional[float] = None
    p99_latency_us: Optional[float] = None
    read_write_ratio: Optional[float] = Field(None, description="Read fraction 0..1")
    io_depth: Optional[float] = Field(None, description="Average I/O queue depth")
    random_ratio: Optional[float] = Field(None, description="Random I/O fraction")


class NetworkCharacteristics(BaseModel):
    """Network I/O profile."""
    packets_per_sec_rx: float = Field(0.0)
    packets_per_sec_tx: float = Field(0.0)
    bandwidth_rx_mbps: float = Field(0.0)
    bandwidth_tx_mbps: float = Field(0.0)
    avg_latency_us: Optional[float] = None
    p99_latency_us: Optional[float] = None
    connection_rate: Optional[float] = Field(None, description="New connections/sec")
    small_packet_ratio: Optional[float] = Field(None, description="Packets < 256B fraction")


class ConcurrencyProfile(BaseModel):
    """Threading and synchronization behavior."""
    thread_count: int = Field(..., description="Active thread count")
    context_switches_per_sec: float = Field(0.0)
    voluntary_cs_ratio: Optional[float] = Field(None,
                                                description="Voluntary CS fraction")
    lock_contention_pct: Optional[float] = Field(None,
                                                 description="Time waiting on locks %")
    futex_wait_time_us: Optional[float] = Field(None,
                                                description="Cumulative futex wait/sec")
    ipc_mechanism: Optional[IPCMechanism] = None


class ScalabilityProfile(BaseModel):
    """Multi-core scaling efficiency."""
    core_counts: List[int] = Field(..., description="Core counts tested")
    throughput_at_core_count: List[float] = Field(..., description="Throughput at each count")
    scaling_efficiency: List[float] = Field(..., description="Efficiency at each count")
    optimal_core_count: Optional[int] = Field(None,
                                              description="Core count where scaling flattens")
    amdahl_serial_fraction: Optional[float] = Field(None,
                                                    description="Serial fraction estimate")


# ---------------------------------------------------------------------------
# Platform configuration models (for optimization analysis)
# ---------------------------------------------------------------------------

class OSConfig(BaseModel):
    """Current OS-level tuning parameters."""
    hugepages_total: int = Field(0, description="vm.nr_hugepages")
    hugepage_size_kb: int = Field(2048)
    transparent_hugepage: str = Field("unknown", description="always/madvise/never")
    cpu_governor: str = Field("unknown", description="scaling_governor value")
    swappiness: int = Field(60, description="vm.swappiness")
    dirty_ratio: int = Field(20, description="vm.dirty_ratio")
    dirty_background_ratio: int = Field(10, description="vm.dirty_background_ratio")
    numa_balancing: bool = Field(True, description="kernel.numa_balancing")
    netdev_max_backlog: int = Field(1000, description="net.core.netdev_max_backlog")
    somaxconn: int = Field(4096, description="net.core.somaxconn")
    tcp_max_syn_backlog: int = Field(1024)
    io_schedulers: Dict[str, str] = Field(
        default_factory=dict, description="device -> scheduler")
    sched_min_granularity_ns: Optional[int] = None
    sched_migration_cost_ns: Optional[int] = None


class BIOSConfig(BaseModel):
    """Current BIOS/firmware-level settings (best-effort detection)."""
    numa_enabled: Optional[bool] = None
    smt_enabled: Optional[bool] = None
    hw_prefetcher_enabled: Optional[bool] = None
    power_profile: str = Field("unknown", description="performance/balanced/powersave")
    c_states_enabled: Optional[bool] = None
    turbo_boost_enabled: Optional[bool] = None


class DriverConfig(BaseModel):
    """Current driver/NIC/storage configuration."""
    nic_offloads: Dict[str, Dict[str, bool]] = Field(
        default_factory=dict, description="interface -> {tso: bool, gro: bool, ...}")
    nic_ring_buffers: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="interface -> {rx: N, tx: N, rx_max: N, tx_max: N}")
    irqbalance_active: bool = Field(True)
    mount_options: Dict[str, List[str]] = Field(
        default_factory=dict, description="mountpoint -> [option, ...]")


class PlatformConfigSnapshot(BaseModel):
    """Complete snapshot of current platform tuning configuration."""
    os: OSConfig = Field(default_factory=OSConfig)
    bios: BIOSConfig = Field(default_factory=BIOSConfig)
    driver: DriverConfig = Field(default_factory=DriverConfig)


# ---------------------------------------------------------------------------
# Power and thermal characteristics
# ---------------------------------------------------------------------------

class PowerThermal(BaseModel):
    """Power consumption and thermal characteristics.

    Data sources:
    - /sys/class/hwmon (hardware monitoring sensors)
    - /sys/class/thermal (thermal zones)
    - /sys/devices/system/cpu/cpu*/cpuidle (C-state residency)
    """
    # Power metrics (watts)
    cpu_power_w: Optional[float] = Field(None, description="CPU package power (watts)")
    gpu_power_w: Optional[float] = Field(None, description="GPU power if present (watts)")
    dram_power_w: Optional[float] = Field(None, description="DRAM power (watts)")
    total_power_w: Optional[float] = Field(None, description="Total system power (watts)")

    # Temperature metrics (celsius)
    cpu_temp_c: Optional[float] = Field(None, description="CPU temperature (celsius)")
    cpu_temp_max_c: Optional[float] = Field(None, description="CPU max temperature threshold")
    dram_temp_c: Optional[float] = Field(None, description="DRAM temperature if available")
    motherboard_temp_c: Optional[float] = Field(None, description="Motherboard/ambient temperature")

    # C-state residency (fraction of time in each state)
    c0_residency: Optional[float] = Field(None, ge=0.0, le=1.0,
                                          description="Fraction of time in C0 (active)")
    c1_residency: Optional[float] = Field(None, ge=0.0, le=1.0,
                                          description="Fraction of time in C1 (halt)")
    c2_residency: Optional[float] = Field(None, ge=0.0, le=1.0,
                                          description="Fraction of time in C2 (stop-clock)")
    c3_residency: Optional[float] = Field(None, ge=0.0, le=1.0,
                                          description="Fraction of time in C3 (deep sleep)")
    c6_residency: Optional[float] = Field(None, ge=0.0, le=1.0,
                                          description="Fraction of time in C6 (deepest)")

    # P-state / frequency
    avg_freq_mhz: Optional[float] = Field(None, description="Average CPU frequency (MHz)")
    min_freq_mhz: Optional[float] = Field(None, description="Minimum frequency observed")
    max_freq_mhz: Optional[float] = Field(None, description="Maximum frequency observed")

    # Thermal throttling
    thermal_throttling_pct: Optional[float] = Field(None, ge=0.0, le=100.0,
                                                    description="Time spent thermally throttled")


# ---------------------------------------------------------------------------
# The unified feature vector
# ---------------------------------------------------------------------------

class WorkloadFeatureVector(BaseModel):
    """The unified feature vector for any workload scenario.

    This is the central data structure of the framework. Every workload
    is distilled into this same schema so chip architects can compare
    MySQL OLTP vs Spark batch vs H.265 encoding along identical axes.
    """

    # -- Metadata --
    scenario_name: str
    scenario_type: ScenarioType
    timestamp: str  # ISO 8601
    platform: str  # e.g. "Kunpeng 920"
    kernel_version: str
    collection_duration_sec: float

    # -- Feature dimensions --
    compute: ComputeCharacteristics
    cache: CacheHierarchy
    branch: BranchBehavior
    memory: MemorySubsystem
    io: IOCharacteristics
    network: NetworkCharacteristics
    concurrency: ConcurrencyProfile
    power_thermal: Optional[PowerThermal] = None
    scalability: Optional[ScalabilityProfile] = None

    # -- Derived (computed by analysis engine) --
    bottleneck_summary: Optional[str] = None
    design_sensitivity: Optional[Dict[str, float]] = None

    # -- Platform config (for optimization analysis) --
    platform_config: Optional[PlatformConfigSnapshot] = None
