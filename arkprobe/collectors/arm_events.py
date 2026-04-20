"""Kunpeng ARM PMU event definitions and grouping strategy.

Kunpeng 920 (TSV110 core) has 6 programmable PMU counters + 1 fixed cycle counter.
Event groups are sized to fit within this limit to avoid multiplexing overhead.

Kunpeng also provides uncore PMUs for DDR controller and L3 cache statistics
that are essential for accurate memory bandwidth and LLC measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
        pmu_name="armv8_pmuv3_0",
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
# Uncore PMU event templates (model-specific)
# ---------------------------------------------------------------------------

# Kunpeng 920: 4 DDRC per SCCL (ddrc0-3), 4 L3C per SCCL (l3c0-3)
UNCORE_EVENTS_920 = {
    "ddr_bandwidth": EventGroup(
        name="ddr_bandwidth",
        description="DDR controller read/write bandwidth",
        events={
            "flux_rd": "hisi_sccl{sccl_id}_ddrc{ddrc_id}/flux_rd/",
            "flux_wr": "hisi_sccl{sccl_id}_ddrc{ddrc_id}/flux_wr/",
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
    ),
}

# Kunpeng 930: 4 DDRC per SCCL (ddrc0_0, ddrc0_1, ddrc2_0, ddrc2_1),
#              10 L3C per SCCL (l3c0-9), 4 HHA per SCCL, 4 SLLC per SCCL
UNCORE_EVENTS_930 = {
    "ddr_bandwidth": EventGroup(
        name="ddr_bandwidth",
        description="DDR controller read/write bandwidth",
        events={
            "flux_rd": "hisi_sccl{sccl_id}_{ddrc_subpath}/flux_rd/",
            "flux_wr": "hisi_sccl{sccl_id}_{ddrc_subpath}/flux_wr/",
        },
    ),
    "l3_cache_uncore": EventGroup(
        name="l3_cache_uncore",
        description="L3 cache from uncore PMU",
        events={
            "dat_access": "hisi_sccl{sccl_id}_l3c{l3c_id}/dat_access/",
            "l3c_hit": "hisi_sccl{sccl_id}_l3c{l3c_id}/l3c_hit/",
            "l3c_ref": "hisi_sccl{sccl_id}_l3c{l3c_id}/l3c_ref/",
        },
    ),
}

# Model-specific SCCL IDs and unit configurations
UNCORE_CONFIG: Dict[str, Dict[str, Any]] = {
    "920": {
        "sccl_ids": [1, 3, 5, 7],
        "ddrc_unit_count": 4,
        "ddrc_subpath": "{ddrc_id}",       # ddrc0, ddrc1, ddrc2, ddrc3
        "l3c_unit_count": 4,
        "l3c_event_type": "920",            # Use 920-style events
    },
    "930": {
        "sccl_ids": [1, 3, 9, 11],
        "ddrc_unit_count": 4,
        "ddrc_subpath": "{ddrc_subpath}",  # ddrc0_0, ddrc0_1, ddrc2_0, ddrc2_1
        "l3c_unit_count": 10,
        "l3c_event_type": "930",           # Use 930-style events
    },
}

# ---------------------------------------------------------------------------
# Kunpeng 930 core PMU event groups
# TaiShan V200 lacks: vfp_spec, ase_spec, ld_spec, st_spec, br_immed_spec,
#   br_indirect_spec, br_return_spec, bus_cycles
# It provides: br_pred, br_mis_pred, br_return_retired, l1d_cache_lmiss_rd,
#   l1i_cache_lmiss, ld_align_lat, st_align_lat
# ---------------------------------------------------------------------------

CORE_EVENT_GROUPS_930: Dict[str, EventGroup] = {
    "topdown_l1": CORE_EVENT_GROUPS["topdown_l1"],
    "instruction_mix": EventGroup(
        name="instruction_mix",
        description="Instruction type breakdown (930-compatible)",
        events={
            "inst_retired": "inst_retired",
            "br_retired": "br_retired",
            "inst_spec": "inst_spec",
        },
        formulas={
            "branch_ratio": "br_retired / inst_retired",
            "speculation_ratio": "inst_spec / inst_retired",
        },
    ),
    "cache_l1": EventGroup(
        name="cache_l1",
        description="L1 instruction and data cache (930 with latency)",
        events={
            "l1d_cache": "l1d_cache",
            "l1d_cache_refill": "l1d_cache_refill",
            "l1d_cache_lmiss_rd": "l1d_cache_lmiss_rd",
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
    "cache_l2_l3": CORE_EVENT_GROUPS["cache_l2_l3"],
    "branch_prediction": EventGroup(
        name="branch_prediction",
        description="Branch prediction accuracy (930-compatible)",
        events={
            "br_retired": "br_retired",
            "br_mis_pred_retired": "br_mis_pred_retired",
            "br_pred": "br_pred",
            "br_mis_pred": "br_mis_pred",
            "br_return_retired": "br_return_retired",
            "inst_retired": "inst_retired",
        },
        formulas={
            "mispredict_rate": "br_mis_pred_retired / br_retired",
            "branch_mpki": "(br_mis_pred_retired / inst_retired) * 1000",
            "pred_accuracy": "br_pred / (br_pred + br_mis_pred)",
            "return_ratio": "br_return_retired / br_retired",
            "branch_density": "br_retired / inst_retired",
        },
    ),
    "memory_access": EventGroup(
        name="memory_access",
        description="Memory access and TLB behavior (930-compatible)",
        events={
            "mem_access": "mem_access",
            "bus_access": "bus_access",
            "dtlb_walk": "dtlb_walk",
            "itlb_walk": "itlb_walk",
            "inst_retired": "inst_retired",
        },
        formulas={
            "tlb_mpki": "((dtlb_walk + itlb_walk) / inst_retired) * 1000",
        },
    ),
}

# Default uncore event groups (Kunpeng 920)
UNCORE_EVENT_GROUPS = UNCORE_EVENTS_920


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_uncore_config(model_id: str) -> Dict[str, Any]:
    """Get uncore PMU configuration for a specific Kunpeng model."""
    if model_id not in UNCORE_CONFIG:
        raise ValueError(
            f"Unknown Kunpeng model for uncore: {model_id}. "
            f"Known: {list(UNCORE_CONFIG)}"
        )
    return UNCORE_CONFIG[model_id]


def get_uncore_event_groups(model_id: str = "920") -> Dict[str, EventGroup]:
    """Get uncore event groups for a specific Kunpeng model."""
    if model_id == "930":
        return UNCORE_EVENTS_930
    return UNCORE_EVENTS_920


def get_kunpeng_model(model_id: str = "920") -> KunpengModel:
    """Get hardware config for a Kunpeng model."""
    if model_id not in KUNPENG_MODELS:
        raise ValueError(f"Unknown Kunpeng model: {model_id}. Known: {list(KUNPENG_MODELS)}")
    return KUNPENG_MODELS[model_id]


def get_all_core_event_groups(model_id: str = "920") -> List[EventGroup]:
    """Return all core PMU event groups in recommended collection order."""
    if model_id == "930":
        groups = CORE_EVENT_GROUPS_930
    else:
        groups = CORE_EVENT_GROUPS
    return [groups[name] for name in [
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
    model_id: str = "920",
) -> List[str]:
    """Resolve uncore event templates with actual SCCL and unit IDs.

    Kunpeng 920 uncore PMUs:
      hisi_sccl3_ddrc0/flux_rd/  (SCCL node 3, DDR controller 0)
      hisi_sccl3_l3c0/rd_hit_cpipe/  (SCCL node 3, L3 cache 0)

    Kunpeng 930 uncore PMUs:
      hisi_sccl3_ddrc0_0/flux_rd/  (SCCL node 3, DDRC sub-row 0, col 0)
      hisi_sccl3_l3c0/dat_access/  (SCCL node 3, L3 cache 0)
    """
    config = get_uncore_config(model_id)
    if sccl_ids is None:
        sccl_ids = config["sccl_ids"]

    resolved: List[str] = []
    for event_str in group.events.values():
        for sccl in sccl_ids:
            if model_id == "930":
                # DDRC uses {ddrc_subpath} placeholder
                if "{ddrc_subpath}" in event_str:
                    ddrc_subpaths = ["ddrc0_0", "ddrc0_1", "ddrc2_0", "ddrc2_1"]
                    for subpath in ddrc_subpaths:
                        resolved.append(
                            event_str.replace("{sccl_id}", str(sccl))
                                     .replace("{ddrc_subpath}", subpath)
                        )
                # L3C uses {l3c_id} — expand 0..9
                if "{l3c_id}" in event_str:
                    for lid in range(config["l3c_unit_count"]):
                        resolved.append(
                            event_str.replace("{sccl_id}", str(sccl))
                                     .replace("{l3c_id}", str(lid))
                        )
            else:
                # Kunpeng 920: DDRC and L3C share same unit index 0-3
                unit_count = config["ddrc_unit_count"]
                for uid in range(unit_count):
                    resolved.append(
                        event_str.replace("{sccl_id}", str(sccl))
                                 .replace("{ddrc_id}", str(uid))
                                 .replace("{l3c_id}", str(uid))
                    )
    return resolved
