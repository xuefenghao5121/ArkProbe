"""Kunpeng ARM PMU event definitions and grouping strategy.

Kunpeng 920 (TSV110 core) has 6 programmable PMU counters + 1 fixed cycle counter.
Event groups are sized to fit within this limit to avoid multiplexing overhead.

Kunpeng also provides uncore PMUs for DDR controller and L3 cache statistics
that are essential for accurate memory bandwidth and LLC measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EventGroup:
    """A group of PMU events that can be collected in a single perf stat run."""
    name: str
    description: str
    events: Dict[str, str]  # logical_name -> perf event string
    formulas: Dict[str, str] = field(default_factory=dict)  # derived_metric -> formula


@dataclass
class KunpengModel:
    """Hardware characteristics for a specific Kunpeng processor model."""
    model: str
    core_name: str
    pmu_name: str
    programmable_counters: int
    dispatch_width: int
    l1d_cache_kb: int
    l1i_cache_kb: int
    l2_cache_kb: int
    l3_cache_mb_per_cluster: int
    memory_channels: int
    max_bandwidth_gbps: float
    supported_features: List[str]


# ---------------------------------------------------------------------------
# Kunpeng processor model definitions
# ---------------------------------------------------------------------------

KUNPENG_MODELS: Dict[str, KunpengModel] = {
    "920": KunpengModel(
        model="920",
        core_name="TSV110",
        pmu_name="armv8_pmuv3",
        programmable_counters=6,
        dispatch_width=4,
        l1d_cache_kb=64,
        l1i_cache_kb=64,
        l2_cache_kb=512,
        l3_cache_mb_per_cluster=32,
        memory_channels=8,
        max_bandwidth_gbps=25.6 * 8,  # 8 channels * 25.6 GB/s DDR4-3200
        supported_features=["neon"],
    ),
    "930": KunpengModel(
        model="930",
        core_name="TaiShan V200",
        pmu_name="armv8_pmuv3",
        programmable_counters=6,
        dispatch_width=8,
        l1d_cache_kb=64,
        l1i_cache_kb=64,
        l2_cache_kb=1024,
        l3_cache_mb_per_cluster=64,
        memory_channels=12,
        max_bandwidth_gbps=38.4 * 12,  # 12 channels DDR5-4800
        supported_features=["neon", "sve"],
    ),
}


# ---------------------------------------------------------------------------
# Core PMU event groups (armv8_pmuv3)
# ---------------------------------------------------------------------------

CORE_EVENT_GROUPS: Dict[str, EventGroup] = {
    "topdown_l1": EventGroup(
        name="topdown_l1",
        description="TopDown Level 1 breakdown",
        events={
            "cycles": "cpu_cycles",
            "instructions": "inst_retired",
            "stall_frontend": "stall_frontend",
            "stall_backend": "stall_backend",
        },
        formulas={
            "frontend_bound": "stall_frontend / cycles",
            "backend_bound": "stall_backend / cycles",
            "retiring": "(instructions / cycles) / DISPATCH_WIDTH",
            "bad_speculation": "1.0 - frontend_bound - backend_bound - retiring",
        },
    ),
    "instruction_mix": EventGroup(
        name="instruction_mix",
        description="Instruction type breakdown",
        events={
            "inst_retired": "inst_retired",
            "br_retired": "br_retired",
            "vfp_spec": "vfp_spec",
            "ase_spec": "ase_spec",
            "ld_spec": "ld_spec",
            "st_spec": "st_spec",
        },
        formulas={
            "branch_ratio": "br_retired / inst_retired",
            "fp_ratio": "vfp_spec / inst_retired",
            "vector_ratio": "ase_spec / inst_retired",
            "load_ratio": "ld_spec / inst_retired",
            "store_ratio": "st_spec / inst_retired",
            "integer_ratio": "1.0 - branch_ratio - fp_ratio - vector_ratio - load_ratio - store_ratio",
        },
    ),
    "cache_l1": EventGroup(
        name="cache_l1",
        description="L1 instruction and data cache",
        events={
            "l1d_cache": "l1d_cache",
            "l1d_cache_refill": "l1d_cache_refill",
            "l1i_cache": "l1i_cache",
            "l1i_cache_refill": "l1i_cache_refill",
            "inst_retired": "inst_retired",
        },
        formulas={
            "l1d_miss_rate": "l1d_cache_refill / l1d_cache",
            "l1d_mpki": "(l1d_cache_refill / inst_retired) * 1000",
            "l1i_miss_rate": "l1i_cache_refill / l1i_cache",
            "l1i_mpki": "(l1i_cache_refill / inst_retired) * 1000",
        },
    ),
    "cache_l2_l3": EventGroup(
        name="cache_l2_l3",
        description="L2 and L3 cache (core-level view)",
        events={
            "l2d_cache": "l2d_cache",
            "l2d_cache_refill": "l2d_cache_refill",
            "l3d_cache": "l3d_cache",
            "l3d_cache_refill": "l3d_cache_refill",
            "inst_retired": "inst_retired",
        },
        formulas={
            "l2_miss_rate": "l2d_cache_refill / l2d_cache",
            "l2_mpki": "(l2d_cache_refill / inst_retired) * 1000",
            "l3_miss_rate": "l3d_cache_refill / l3d_cache",
            "l3_mpki": "(l3d_cache_refill / inst_retired) * 1000",
        },
    ),
    "branch_prediction": EventGroup(
        name="branch_prediction",
        description="Branch prediction accuracy",
        events={
            "br_retired": "br_retired",
            "br_mis_pred_retired": "br_mis_pred_retired",
            "br_immed_spec": "br_immed_spec",
            "br_indirect_spec": "br_indirect_spec",
            "br_return_spec": "br_return_spec",
            "inst_retired": "inst_retired",
        },
        formulas={
            "mispredict_rate": "br_mis_pred_retired / br_retired",
            "branch_mpki": "(br_mis_pred_retired / inst_retired) * 1000",
            "indirect_ratio": "br_indirect_spec / (br_immed_spec + br_indirect_spec + br_return_spec)",
            "branch_density": "br_retired / inst_retired",
        },
    ),
    "memory_access": EventGroup(
        name="memory_access",
        description="Memory access and TLB behavior",
        events={
            "mem_access": "mem_access",
            "bus_access": "bus_access",
            "bus_cycles": "bus_cycles",
            "dtlb_walk": "dtlb_walk",
            "itlb_walk": "itlb_walk",
            "inst_retired": "inst_retired",
        },
        formulas={
            "tlb_mpki": "((dtlb_walk + itlb_walk) / inst_retired) * 1000",
            "bus_utilization": "bus_access / bus_cycles",
        },
    ),
}

# ---------------------------------------------------------------------------
# Uncore PMU events (Kunpeng-specific)
# ---------------------------------------------------------------------------

UNCORE_EVENT_GROUPS: Dict[str, EventGroup] = {
    "ddr_bandwidth": EventGroup(
        name="ddr_bandwidth",
        description="DDR controller read/write bandwidth",
        events={
            "flux_rd": "hisi_sccl{sccl_id}_ddrc{ddrc_id}/flux_rd/",
            "flux_wr": "hisi_sccl{sccl_id}_ddrc{ddrc_id}/flux_wr/",
        },
        formulas={
            "read_bandwidth_gbps": "(flux_rd * 32) / (duration_sec * 1e9)",
            "write_bandwidth_gbps": "(flux_wr * 32) / (duration_sec * 1e9)",
        },
    ),
    "l3_cache_uncore": EventGroup(
        name="l3_cache_uncore",
        description="L3 cache hit/miss from uncore PMU",
        events={
            "rd_hit_cpipe": "hisi_sccl{sccl_id}_l3c{l3c_id}/rd_hit_cpipe/",
            "rd_miss_cpipe": "hisi_sccl{sccl_id}_l3c{l3c_id}/rd_miss_cpipe/",
            "wr_hit_cpipe": "hisi_sccl{sccl_id}_l3c{l3c_id}/wr_hit_cpipe/",
            "wr_miss_cpipe": "hisi_sccl{sccl_id}_l3c{l3c_id}/wr_miss_cpipe/",
        },
        formulas={
            "l3_hit_rate": "(rd_hit_cpipe + wr_hit_cpipe) / (rd_hit_cpipe + rd_miss_cpipe + wr_hit_cpipe + wr_miss_cpipe)",
        },
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_kunpeng_model(model_id: str = "920") -> KunpengModel:
    """Get hardware config for a Kunpeng model."""
    if model_id not in KUNPENG_MODELS:
        raise ValueError(f"Unknown Kunpeng model: {model_id}. Known: {list(KUNPENG_MODELS)}")
    return KUNPENG_MODELS[model_id]


def get_all_core_event_groups() -> List[EventGroup]:
    """Return all core PMU event groups in recommended collection order."""
    return [CORE_EVENT_GROUPS[name] for name in [
        "topdown_l1", "instruction_mix", "cache_l1",
        "cache_l2_l3", "branch_prediction", "memory_access",
    ]]


def build_perf_event_string(group: EventGroup, pmu_name: str = "armv8_pmuv3") -> str:
    """Build the -e argument for perf stat from an event group.

    Returns comma-separated event specifiers like:
    'armv8_pmuv3/cpu_cycles/,armv8_pmuv3/inst_retired/,...'
    """
    parts = []
    for event_str in group.events.values():
        if "/" in event_str:
            # Already fully qualified (uncore events)
            parts.append(event_str)
        else:
            parts.append(f"{pmu_name}/{event_str}/")
    return ",".join(parts)


def resolve_uncore_events(
    group: EventGroup,
    sccl_ids: Optional[List[int]] = None,
    unit_ids: Optional[List[int]] = None,
) -> List[str]:
    """Resolve uncore event templates with actual SCCL and unit IDs.

    Kunpeng uncore PMUs are named like:
    hisi_sccl3_ddrc0/flux_rd/  (SCCL node 3, DDR controller 0)
    hisi_sccl3_l3c0/rd_hit_cpipe/  (SCCL node 3, L3 cache 0)
    """
    if sccl_ids is None:
        sccl_ids = [1, 3, 5, 7]  # Default 4-socket Kunpeng 920
    if unit_ids is None:
        unit_ids = [0, 1, 2, 3]  # 4 units per SCCL

    resolved = []
    for event_str in group.events.values():
        for sccl in sccl_ids:
            for uid in unit_ids:
                resolved.append(
                    event_str.replace("{sccl_id}", str(sccl))
                             .replace("{ddrc_id}", str(uid))
                             .replace("{l3c_id}", str(uid))
                )
    return resolved
