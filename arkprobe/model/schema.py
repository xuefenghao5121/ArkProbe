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
    scalability: Optional[ScalabilityProfile] = None

    # -- Derived (computed by analysis engine) --
    bottleneck_summary: Optional[str] = None
    design_sensitivity: Optional[Dict[str, float]] = None
