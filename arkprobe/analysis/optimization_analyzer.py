"""Platform optimization analyzer — OS / BIOS / Driver tuning recommendations.

Maps workload feature vectors + current platform configuration to actionable
optimization recommendations. Each recommendation includes impact scoring,
gap detection (current vs recommended), and executable commands.

Tuning rules are differentiated by scenario type (database, codec, microservice, etc.)
and modulated by feature vector metrics for precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..model.enums import ScenarioType, TuningDifficulty, TuningLayer, TuningRiskLevel
from ..model.schema import PlatformConfigSnapshot, WorkloadFeatureVector
from ..utils.units import clamp

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TuningRule:
    """A single tuning rule in the knowledge base."""
    parameter_name: str
    display_name: str
    layer: TuningLayer
    difficulty: TuningDifficulty
    risk: TuningRiskLevel
    description: str
    # Recommended values per scenario type (key = ScenarioType.value or "_default")
    recommended_values: Dict[str, str]
    # Base impact score per scenario type
    base_impact: Dict[str, float]
    # Feature conditions that boost impact: [{metric, op, threshold, boost}]
    impact_conditions: List[Dict[str, Any]] = field(default_factory=list)
    # Prerequisites: rule is suppressed unless ALL conditions are met
    # e.g. [{"metric": "io.iops_write", "op": ">", "threshold": 0}]
    prerequisites: List[Dict[str, Any]] = field(default_factory=list)
    # Command templates
    apply_template: str = ""
    verify_template: str = ""
    rollback_template: str = ""
    # Config snapshot path (dot-separated, e.g. "os.swappiness")
    config_path: str = ""


@dataclass
class OptimizationRecommendation:
    """A single optimization recommendation with gap analysis."""
    parameter_name: str
    display_name: str
    layer: TuningLayer
    difficulty: TuningDifficulty
    risk: TuningRiskLevel
    description: str
    current_value: str
    recommended_value: str
    impact_score: float        # 0.0 - 1.0
    priority_score: float      # impact * ease_multiplier
    gap_detected: Optional[bool]  # None if current unknown
    reasoning: str
    apply_commands: List[str]
    verify_commands: List[str]
    rollback_commands: List[str]


@dataclass
class LayerSummary:
    """Summary of recommendations for one tuning layer."""
    layer: TuningLayer
    total_parameters: int
    gaps_found: int
    max_impact: float
    recommendations: List[OptimizationRecommendation]


@dataclass
class OptimizationReport:
    """Complete optimization analysis for one workload scenario."""
    scenario_name: str
    scenario_type: ScenarioType
    layers: Dict[str, LayerSummary]
    all_recommendations: List[OptimizationRecommendation]
    optimization_score: float   # 0-100, higher = better tuned
    summary: str


@dataclass
class CrossScenarioOptimizationReport:
    """Optimization analysis across multiple workloads."""
    per_scenario: Dict[str, OptimizationReport]
    universal_recommendations: List[OptimizationRecommendation]
    conflicting_parameters: List[Dict[str, Any]]
    parameter_benefit_matrix: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Difficulty multipliers for priority scoring
# ---------------------------------------------------------------------------

DIFFICULTY_MULTIPLIER = {
    TuningDifficulty.TRIVIAL: 1.0,
    TuningDifficulty.EASY: 0.8,
    TuningDifficulty.MODERATE: 0.5,
    TuningDifficulty.HARD: 0.3,
}


# ---------------------------------------------------------------------------
# Tuning knowledge base
# ---------------------------------------------------------------------------

TUNING_RULES: List[TuningRule] = [
    # ===== OS Layer =====
    # OS rules irrelevant to compute_bound / memory_bound / mixed are excluded
    # via base_impact=0.0 so they won't appear in recommendations
    TuningRule(
        parameter_name="vm.nr_hugepages",
        display_name="Huge Pages Count",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Pre-allocate huge pages to reduce TLB misses for memory-intensive workloads.",
        recommended_values={
            "database_oltp": "auto",
            "database_olap": "auto",
            "database_kv": "auto",
            "bigdata_batch": "1024",
            "bigdata_streaming": "512",
            "codec_video": "512",
            "codec_audio": "256",
            "search_recommend": "1024",
            "microservice": "0",
            "memory_bound": "1024",
            "mixed": "512",
            "compute_bound": "0",
            "_default": "512",
        },
        base_impact={
            "database_oltp": 0.7, "database_olap": 0.5, "database_kv": 0.8,
            "bigdata_batch": 0.4, "bigdata_streaming": 0.3,
            "codec_video": 0.3, "codec_audio": 0.2,
            "search_recommend": 0.4, "microservice": 0.1,
            "memory_bound": 0.6, "mixed": 0.3, "compute_bound": 0.0,
            "_default": 0.3,
        },
        impact_conditions=[
            {"metric": "memory.tlb_mpki", "op": ">", "threshold": 1.0, "boost": 0.2},
            {"metric": "cache.l3_mpki", "op": ">", "threshold": 5.0, "boost": 0.1},
        ],
        prerequisites=[
            {"metric": "memory.tlb_mpki", "op": ">", "threshold": 0.5},
        ],
        apply_template="sysctl -w vm.nr_hugepages={value}",
        verify_template="cat /proc/meminfo | grep HugePages",
        rollback_template="sysctl -w vm.nr_hugepages={current}",
        config_path="os.hugepages_total",
    ),
    TuningRule(
        parameter_name="transparent_hugepage",
        display_name="Transparent Huge Pages",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.MEDIUM,
        description="THP mode: 'always' boosts batch throughput, 'madvise' avoids latency spikes for DB, 'never' for latency-critical.",
        recommended_values={
            "database_oltp": "madvise",
            "database_olap": "always",
            "database_kv": "never",
            "bigdata_batch": "always",
            "bigdata_streaming": "madvise",
            "codec_video": "always",
            "codec_audio": "always",
            "search_recommend": "madvise",
            "microservice": "madvise",
            "memory_bound": "always",
            "mixed": "madvise",
            "compute_bound": "madvise",
            "_default": "madvise",
        },
        base_impact={
            "database_oltp": 0.6, "database_olap": 0.3, "database_kv": 0.7,
            "bigdata_batch": 0.3,
            "memory_bound": 0.4, "mixed": 0.2, "compute_bound": 0.1,
            "_default": 0.3,
        },
        impact_conditions=[
            {"metric": "memory.tlb_mpki", "op": ">", "threshold": 0.5, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "memory.tlb_mpki", "op": ">", "threshold": 0.3},
        ],
        apply_template="echo {value} > /sys/kernel/mm/transparent_hugepage/enabled",
        verify_template="cat /sys/kernel/mm/transparent_hugepage/enabled",
        rollback_template="echo {current} > /sys/kernel/mm/transparent_hugepage/enabled",
        config_path="os.transparent_hugepage",
    ),
    TuningRule(
        parameter_name="vm.swappiness",
        display_name="VM Swappiness",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Controls kernel tendency to swap memory pages. Lower values keep data in RAM.",
        recommended_values={
            "database_oltp": "1",
            "database_olap": "10",
            "database_kv": "1",
            "bigdata_batch": "10",
            "bigdata_streaming": "10",
            "codec_video": "10",
            "codec_audio": "10",
            "search_recommend": "10",
            "microservice": "30",
            "compute_bound": "1",
            "memory_bound": "1",
            "mixed": "10",
            "_default": "10",
        },
        base_impact={
            "database_oltp": 0.7, "database_olap": 0.4, "database_kv": 0.8,
            "bigdata_batch": 0.3,
            "compute_bound": 0.0, "memory_bound": 0.5, "mixed": 0.2,
            "_default": 0.3,
        },
        impact_conditions=[
            {"metric": "memory.bandwidth_utilization", "op": ">", "threshold": 0.5, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "memory.bandwidth_utilization", "op": ">", "threshold": 0.1},
        ],
        apply_template="sysctl -w vm.swappiness={value}",
        verify_template="sysctl vm.swappiness",
        rollback_template="sysctl -w vm.swappiness={current}",
        config_path="os.swappiness",
    ),
    TuningRule(
        parameter_name="vm.dirty_ratio",
        display_name="Dirty Page Ratio",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Max percentage of memory with dirty pages before forced writeback. Lower = more predictable IO latency.",
        recommended_values={
            "database_oltp": "5",
            "database_olap": "10",
            "database_kv": "5",
            "compute_bound": "20",
            "memory_bound": "10",
            "mixed": "20",
            "_default": "20",
        },
        base_impact={
            "database_oltp": 0.5, "database_olap": 0.3, "database_kv": 0.5,
            "compute_bound": 0.0, "memory_bound": 0.1, "mixed": 0.0,
            "_default": 0.15,
        },
        impact_conditions=[
            {"metric": "io.iops_write", "op": ">", "threshold": 1000, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "io.iops_write", "op": ">", "threshold": 0},
        ],
        apply_template="sysctl -w vm.dirty_ratio={value}",
        verify_template="sysctl vm.dirty_ratio",
        rollback_template="sysctl -w vm.dirty_ratio={current}",
        config_path="os.dirty_ratio",
    ),
    TuningRule(
        parameter_name="vm.dirty_background_ratio",
        display_name="Dirty Background Ratio",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Percentage of memory where background writeback starts.",
        recommended_values={
            "database_oltp": "3",
            "database_olap": "5",
            "database_kv": "3",
            "compute_bound": "10",
            "memory_bound": "5",
            "mixed": "10",
            "_default": "10",
        },
        base_impact={
            "database_oltp": 0.4, "database_olap": 0.2, "database_kv": 0.4,
            "compute_bound": 0.0, "memory_bound": 0.1, "mixed": 0.0,
            "_default": 0.1,
        },
        prerequisites=[
            {"metric": "io.iops_write", "op": ">", "threshold": 0},
        ],
        apply_template="sysctl -w vm.dirty_background_ratio={value}",
        verify_template="sysctl vm.dirty_background_ratio",
        rollback_template="sysctl -w vm.dirty_background_ratio={current}",
        config_path="os.dirty_background_ratio",
    ),
    TuningRule(
        parameter_name="scaling_governor",
        display_name="CPU Frequency Governor",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="CPU frequency scaling policy. 'performance' locks max frequency for consistent throughput.",
        recommended_values={
            "database_oltp": "performance",
            "database_olap": "performance",
            "database_kv": "performance",
            "bigdata_batch": "performance",
            "bigdata_streaming": "performance",
            "codec_video": "performance",
            "codec_audio": "performance",
            "search_recommend": "performance",
            "microservice": "ondemand",
            "compute_bound": "performance",
            "memory_bound": "performance",
            "mixed": "performance",
            "_default": "performance",
        },
        base_impact={
            "database_oltp": 0.6, "codec_video": 0.5,
            "microservice": 0.2,
            "compute_bound": 0.7, "memory_bound": 0.5, "mixed": 0.5,
            "_default": 0.4,
        },
        apply_template="cpupower frequency-set -g {value}",
        verify_template="cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        rollback_template="cpupower frequency-set -g {current}",
        config_path="os.cpu_governor",
    ),
    TuningRule(
        parameter_name="kernel.numa_balancing",
        display_name="Automatic NUMA Balancing",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Kernel auto-migration of pages between NUMA nodes. Disable for manually-bound DB workloads.",
        recommended_values={
            "database_oltp": "0",
            "database_kv": "0",
            "memory_bound": "0",
            "mixed": "1",
            "compute_bound": "1",
            "_default": "1",
        },
        base_impact={
            "database_oltp": 0.5, "database_kv": 0.5,
            "memory_bound": 0.4, "mixed": 0.1, "compute_bound": 0.0,
            "_default": 0.15,
        },
        impact_conditions=[
            {"metric": "memory.numa_local_ratio", "op": "<", "threshold": 0.9, "boost": 0.2},
        ],
        apply_template="sysctl -w kernel.numa_balancing={value}",
        verify_template="sysctl kernel.numa_balancing",
        rollback_template="sysctl -w kernel.numa_balancing={current}",
        config_path="os.numa_balancing",
    ),
    TuningRule(
        parameter_name="net.core.netdev_max_backlog",
        display_name="Network Device Backlog",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Max queued packets when interface receives faster than kernel processes.",
        recommended_values={
            "microservice": "65536",
            "search_recommend": "65536",
            "compute_bound": "1000",
            "memory_bound": "1000",
            "mixed": "1000",
            "_default": "1000",
        },
        base_impact={
            "microservice": 0.5, "search_recommend": 0.4,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 100000, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 0},
        ],
        apply_template="sysctl -w net.core.netdev_max_backlog={value}",
        verify_template="sysctl net.core.netdev_max_backlog",
        rollback_template="sysctl -w net.core.netdev_max_backlog={current}",
        config_path="os.netdev_max_backlog",
    ),
    TuningRule(
        parameter_name="net.core.somaxconn",
        display_name="Socket Max Connections",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Max listen backlog for sockets. Increase for high connection rate services.",
        recommended_values={
            "database_oltp": "65535",
            "microservice": "65535",
            "search_recommend": "65535",
            "compute_bound": "4096",
            "memory_bound": "4096",
            "mixed": "4096",
            "_default": "4096",
        },
        base_impact={
            "database_oltp": 0.3, "microservice": 0.5, "search_recommend": 0.4,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "network.connection_rate", "op": ">", "threshold": 1000, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 0},
        ],
        apply_template="sysctl -w net.core.somaxconn={value}",
        verify_template="sysctl net.core.somaxconn",
        rollback_template="sysctl -w net.core.somaxconn={current}",
        config_path="os.somaxconn",
    ),
    TuningRule(
        parameter_name="net.ipv4.tcp_max_syn_backlog",
        display_name="TCP SYN Backlog",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Max pending TCP SYN connections. Increase for high connection rate.",
        recommended_values={
            "microservice": "65535",
            "search_recommend": "65535",
            "compute_bound": "1024",
            "memory_bound": "1024",
            "mixed": "1024",
            "_default": "1024",
        },
        base_impact={
            "microservice": 0.4, "search_recommend": 0.3,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.1,
        },
        prerequisites=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 0},
        ],
        apply_template="sysctl -w net.ipv4.tcp_max_syn_backlog={value}",
        verify_template="sysctl net.ipv4.tcp_max_syn_backlog",
        rollback_template="sysctl -w net.ipv4.tcp_max_syn_backlog={current}",
        config_path="os.tcp_max_syn_backlog",
    ),
    TuningRule(
        parameter_name="io_scheduler",
        display_name="I/O Scheduler",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Block device I/O scheduler. 'mq-deadline' for DB fairness, 'none' for NVMe/SSD.",
        recommended_values={
            "database_oltp": "mq-deadline",
            "database_olap": "mq-deadline",
            "database_kv": "none",
            "compute_bound": "none",
            "memory_bound": "none",
            "mixed": "none",
            "_default": "none",
        },
        base_impact={
            "database_oltp": 0.4, "database_olap": 0.3, "database_kv": 0.3,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.15,
        },
        impact_conditions=[
            {"metric": "io.iops_read", "op": ">", "threshold": 5000, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "io.iops_read", "op": ">", "threshold": 0},
        ],
        apply_template="echo {value} > /sys/block/sda/queue/scheduler",
        verify_template="cat /sys/block/sda/queue/scheduler",
        rollback_template="echo {current} > /sys/block/sda/queue/scheduler",
        config_path="os.io_schedulers",
    ),
    TuningRule(
        parameter_name="sched_min_granularity_ns",
        display_name="Scheduler Min Granularity",
        layer=TuningLayer.OS,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.MEDIUM,
        description="Minimum scheduling time slice. Larger value reduces context switch overhead.",
        recommended_values={
            "database_oltp": "10000000",
            "database_kv": "10000000",
            "compute_bound": "3000000",
            "memory_bound": "3000000",
            "mixed": "3000000",
            "_default": "3000000",
        },
        base_impact={
            "database_oltp": 0.3, "database_kv": 0.3,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "concurrency.context_switches_per_sec", "op": ">", "threshold": 50000, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "concurrency.context_switches_per_sec", "op": ">", "threshold": 1000},
        ],
        apply_template="sysctl -w kernel.sched_min_granularity_ns={value}",
        verify_template="sysctl kernel.sched_min_granularity_ns",
        rollback_template="sysctl -w kernel.sched_min_granularity_ns={current}",
        config_path="os.sched_min_granularity_ns",
    ),

    # ===== BIOS Layer =====
    TuningRule(
        parameter_name="numa_enabled",
        display_name="NUMA Topology",
        layer=TuningLayer.BIOS,
        difficulty=TuningDifficulty.HARD,
        risk=TuningRiskLevel.LOW,
        description="Enable NUMA for multi-socket systems to optimize memory locality.",
        recommended_values={"_default": "True"},
        base_impact={"_default": 0.6, "compute_bound": 0.1, "memory_bound": 0.4, "mixed": 0.2},
        impact_conditions=[
            {"metric": "memory.numa_local_ratio", "op": "<", "threshold": 0.8, "boost": 0.2},
        ],
        apply_template="# BIOS: Enable NUMA (requires BIOS/UEFI configuration)",
        verify_template="numactl --hardware | head -5",
        rollback_template="# BIOS: Revert NUMA setting",
        config_path="bios.numa_enabled",
    ),
    TuningRule(
        parameter_name="hw_prefetcher",
        display_name="Hardware Prefetcher",
        layer=TuningLayer.BIOS,
        difficulty=TuningDifficulty.HARD,
        risk=TuningRiskLevel.LOW,
        description="Enable hardware prefetcher for streaming/sequential memory access patterns.",
        recommended_values={"_default": "True"},
        base_impact={
            "_default": 0.4,
            "compute_bound": 0.5, "memory_bound": 0.6, "mixed": 0.5,
        },
        impact_conditions=[
            {"metric": "cache.spatial_locality_score", "op": ">", "threshold": 0.5, "boost": 0.2},
            {"metric": "cache.l2_mpki", "op": ">", "threshold": 3.0, "boost": 0.15},
        ],
        apply_template="# BIOS: Enable HW Prefetcher (requires BIOS/UEFI configuration)",
        verify_template="# Check /sys/devices/system/cpu/cpu0/prefetch if available",
        rollback_template="# BIOS: Revert HW Prefetcher setting",
        config_path="bios.hw_prefetcher_enabled",
    ),
    TuningRule(
        parameter_name="smt_enabled",
        display_name="Simultaneous Multi-Threading",
        layer=TuningLayer.BIOS,
        difficulty=TuningDifficulty.MODERATE,
        risk=TuningRiskLevel.MEDIUM,
        description="SMT shares core resources. Disable for latency-sensitive DB to avoid resource contention.",
        recommended_values={
            "database_oltp": "False",
            "database_kv": "False",
            "compute_bound": "True",
            "memory_bound": "True",
            "mixed": "True",
            "_default": "True",
        },
        base_impact={
            "database_oltp": 0.4, "database_kv": 0.4,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.2,
        },
        impact_conditions=[
            {"metric": "concurrency.lock_contention_pct", "op": ">", "threshold": 5, "boost": 0.15},
        ],
        apply_template="echo off > /sys/devices/system/cpu/smt/control",
        verify_template="cat /sys/devices/system/cpu/smt/active",
        rollback_template="echo on > /sys/devices/system/cpu/smt/control",
        config_path="bios.smt_enabled",
    ),
    TuningRule(
        parameter_name="power_profile",
        display_name="Power Management Profile",
        layer=TuningLayer.BIOS,
        difficulty=TuningDifficulty.MODERATE,
        risk=TuningRiskLevel.LOW,
        description="Set to 'performance' for consistent throughput, avoid frequency throttling.",
        recommended_values={"_default": "performance"},
        base_impact={
            "_default": 0.5,
            "compute_bound": 0.7, "memory_bound": 0.5, "mixed": 0.5,
        },
        apply_template="# BIOS: Set Power Profile to Max Performance",
        verify_template="cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference",
        rollback_template="# BIOS: Revert Power Profile",
        config_path="bios.power_profile",
    ),
    TuningRule(
        parameter_name="c_states",
        display_name="CPU C-States",
        layer=TuningLayer.BIOS,
        difficulty=TuningDifficulty.MODERATE,
        risk=TuningRiskLevel.MEDIUM,
        description="Disable deep C-states to reduce wakeup latency jitter. Important for latency-sensitive workloads.",
        recommended_values={
            "database_oltp": "False",
            "database_kv": "False",
            "codec_video": "False",
            "codec_audio": "False",
            "microservice": "True",
            "compute_bound": "False",
            "memory_bound": "False",
            "mixed": "False",
            "_default": "False",
        },
        base_impact={
            "database_oltp": 0.5, "database_kv": 0.5,
            "codec_video": 0.3, "microservice": 0.1,
            "compute_bound": 0.4, "memory_bound": 0.3, "mixed": 0.3,
            "_default": 0.3,
        },
        apply_template="# Disable C-states via GRUB: add 'processor.max_cstate=1 intel_idle.max_cstate=0' to GRUB_CMDLINE_LINUX",
        verify_template="cat /sys/devices/system/cpu/cpu0/cpuidle/state*/disable",
        rollback_template="# Re-enable C-states: remove GRUB parameters and reboot",
        config_path="bios.c_states_enabled",
    ),

    # ===== Driver Layer =====
    TuningRule(
        parameter_name="nic_tso_gro",
        display_name="NIC Offload (TSO/GRO)",
        layer=TuningLayer.DRIVER,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Enable TCP segmentation offload and generic receive offload for network throughput.",
        recommended_values={
            "microservice": "on",
            "search_recommend": "on",
            "compute_bound": "default",
            "memory_bound": "default",
            "mixed": "default",
            "_default": "on",
        },
        base_impact={
            "microservice": 0.5, "search_recommend": 0.4,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.2,
        },
        impact_conditions=[
            {"metric": "network.bandwidth_rx_mbps", "op": ">", "threshold": 1000, "boost": 0.2},
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 50000, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 0},
        ],
        apply_template="ethtool -K eth0 tso on gro on",
        verify_template="ethtool -k eth0 | grep -E 'tcp-segmentation|generic-receive'",
        rollback_template="ethtool -K eth0 tso off gro off",
        config_path="driver.nic_offloads",
    ),
    TuningRule(
        parameter_name="nic_ring_buffer",
        display_name="NIC Ring Buffer Size",
        layer=TuningLayer.DRIVER,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Maximize NIC ring buffers to handle packet bursts without drops.",
        recommended_values={
            "microservice": "max",
            "search_recommend": "max",
            "compute_bound": "default",
            "memory_bound": "default",
            "mixed": "default",
            "_default": "default",
        },
        base_impact={
            "microservice": 0.4, "search_recommend": 0.3,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 100000, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 0},
        ],
        apply_template="ethtool -G eth0 rx {value} tx {value}",
        verify_template="ethtool -g eth0",
        rollback_template="ethtool -G eth0 rx {current} tx {current}",
        config_path="driver.nic_ring_buffers",
    ),
    TuningRule(
        parameter_name="irqbalance",
        display_name="IRQ Balancing",
        layer=TuningLayer.DRIVER,
        difficulty=TuningDifficulty.EASY,
        risk=TuningRiskLevel.MEDIUM,
        description="irqbalance distributes interrupts across CPUs. Disable for DB with manual CPU affinity.",
        recommended_values={
            "database_oltp": "False",
            "database_kv": "False",
            "compute_bound": "True",
            "memory_bound": "True",
            "mixed": "True",
            "_default": "True",
        },
        base_impact={
            "database_oltp": 0.3, "database_kv": 0.3,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.1,
        },
        apply_template="systemctl stop irqbalance && systemctl disable irqbalance",
        verify_template="systemctl is-active irqbalance",
        rollback_template="systemctl enable irqbalance && systemctl start irqbalance",
        config_path="driver.irqbalance_active",
    ),
    TuningRule(
        parameter_name="fs_noatime",
        display_name="Filesystem noatime",
        layer=TuningLayer.DRIVER,
        difficulty=TuningDifficulty.EASY,
        risk=TuningRiskLevel.LOW,
        description="Disable access time updates on filesystem to reduce write IO for DB workloads.",
        recommended_values={
            "database_oltp": "noatime",
            "database_olap": "noatime",
            "database_kv": "noatime",
            "compute_bound": "default",
            "memory_bound": "default",
            "mixed": "default",
            "_default": "default",
        },
        base_impact={
            "database_oltp": 0.4, "database_olap": 0.3, "database_kv": 0.4,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "io.iops_write", "op": ">", "threshold": 1000, "boost": 0.1},
        ],
        prerequisites=[
            {"metric": "io.iops_write", "op": ">", "threshold": 0},
        ],
        apply_template="mount -o remount,noatime /data",
        verify_template="mount | grep /data",
        rollback_template="mount -o remount,relatime /data",
        config_path="driver.mount_options",
    ),
    TuningRule(
        parameter_name="rps_rfs",
        display_name="Receive Packet Steering (RPS/RFS)",
        layer=TuningLayer.DRIVER,
        difficulty=TuningDifficulty.TRIVIAL,
        risk=TuningRiskLevel.LOW,
        description="Distribute network packet processing across CPUs for high packet rate scenarios.",
        recommended_values={
            "microservice": "on",
            "search_recommend": "on",
            "compute_bound": "off",
            "memory_bound": "off",
            "mixed": "off",
            "_default": "off",
        },
        base_impact={
            "microservice": 0.4, "search_recommend": 0.3,
            "compute_bound": 0.0, "memory_bound": 0.0, "mixed": 0.0,
            "_default": 0.05,
        },
        impact_conditions=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 200000, "boost": 0.25},
        ],
        prerequisites=[
            {"metric": "network.packets_per_sec_rx", "op": ">", "threshold": 0},
        ],
        apply_template="echo ffff > /sys/class/net/eth0/queues/rx-0/rps_cpus",
        verify_template="cat /sys/class/net/eth0/queues/rx-0/rps_cpus",
        rollback_template="echo 0 > /sys/class/net/eth0/queues/rx-0/rps_cpus",
        config_path="driver.rps_rfs",
    ),

    # ===== JVM Layer =====
    TuningRule(
        parameter_name="jvm.heap_max_size",
        display_name="JVM Max Heap Size",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.MODERATE,
        risk=TuningRiskLevel.MEDIUM,
        description="Maximum Java heap size. Too small causes frequent GC; too large increases GC pause times.",
        recommended_values={
            "jvm_general": "75%_physical",
            "jvm_gc_heavy": "70%_physical",
            "jvm_jit_intensive": "60%_physical",
            "bigdata_batch": "80%_physical",
            "database_oltp": "50%_physical",
            "microservice": "60%_physical",
            "_default": "70%_physical",
        },
        base_impact={
            "jvm_general": 0.5, "jvm_gc_heavy": 0.8, "jvm_jit_intensive": 0.3,
            "bigdata_batch": 0.5, "database_oltp": 0.4, "microservice": 0.3,
            "_default": 0.3,
        },
        impact_conditions=[
            {"metric": "jvm.gc.gc_pause_ratio", "op": ">", "threshold": 0.05, "boost": 0.2},
            {"metric": "jvm.gc.heap_usage_ratio", "op": ">", "threshold": 0.80, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "jvm.gc.heap_usage_ratio", "op": ">", "threshold": 0.0},
        ],
        apply_template="Restart JVM with -Xmx{value}",
        verify_template="jcmd {pid} VM.flags | grep MaxHeapSize",
        rollback_template="Restart JVM with original -Xmx",
        config_path="jvm.heap_max_size",
    ),
    TuningRule(
        parameter_name="jvm.gc_algorithm",
        display_name="JVM Garbage Collector",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.MODERATE,
        risk=TuningRiskLevel.MEDIUM,
        description="GC algorithm selection. ZGC for low latency, G1 for balanced, Parallel for max throughput.",
        recommended_values={
            "jvm_general": "G1",
            "jvm_gc_heavy": "ZGC",
            "jvm_jit_intensive": "G1",
            "database_oltp": "ZGC",
            "database_kv": "ZGC",
            "bigdata_batch": "Parallel",
            "bigdata_streaming": "G1",
            "microservice": "G1",
            "_default": "G1",
        },
        base_impact={
            "jvm_general": 0.4, "jvm_gc_heavy": 0.9, "jvm_jit_intensive": 0.2,
            "database_oltp": 0.7, "database_kv": 0.6, "bigdata_batch": 0.5,
            "_default": 0.3,
        },
        impact_conditions=[
            {"metric": "jvm.gc.gc_pause_ratio", "op": ">", "threshold": 0.10, "boost": 0.3},
            {"metric": "jvm.gc.full_gc_count", "op": ">", "threshold": 0, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "jvm.gc.gc_pause_ratio", "op": ">", "threshold": 0.03},
        ],
        apply_template="Restart JVM with -XX:+Use{value}GC",
        verify_template="jcmd {pid} VM.flags | grep Use.*GC",
        rollback_template="Restart JVM with original GC flag",
        config_path="jvm.gc_algorithm",
    ),
    TuningRule(
        parameter_name="jvm.gc_thread_count",
        display_name="JVM GC Thread Count",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.EASY,
        risk=TuningRiskLevel.LOW,
        description="Number of parallel GC threads. Default is usually CPU count, but may need tuning for NUMA.",
        recommended_values={
            "jvm_general": "cpu_count",
            "jvm_gc_heavy": "cpu_count/2",
            "bigdata_batch": "cpu_count",
            "_default": "cpu_count",
        },
        base_impact={
            "jvm_general": 0.2, "jvm_gc_heavy": 0.5,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "jvm.gc.gc_pause_ratio", "op": ">", "threshold": 0.05, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "jvm.gc.gc_pause_ratio", "op": ">", "threshold": 0.03},
        ],
        apply_template="Restart JVM with -XX:ParallelGCThreads={value}",
        verify_template="jcmd {pid} VM.flags | grep ParallelGCThreads",
        rollback_template="Restart JVM without -XX:ParallelGCThreads override",
        config_path="jvm.gc_thread_count",
    ),
    TuningRule(
        parameter_name="jvm.young_gen_size",
        display_name="JVM Young Generation Size",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.MODERATE,
        risk=TuningRiskLevel.MEDIUM,
        description="Young generation heap size. Larger young gen reduces young GC frequency but increases pause time.",
        recommended_values={
            "jvm_general": "heap/3",
            "jvm_gc_heavy": "heap/2",
            "jvm_jit_intensive": "heap/4",
            "_default": "heap/3",
        },
        base_impact={
            "jvm_general": 0.3, "jvm_gc_heavy": 0.6, "jvm_jit_intensive": 0.1,
            "_default": 0.2,
        },
        impact_conditions=[
            {"metric": "jvm.gc.young_gc_count", "op": ">", "threshold": 10, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "jvm.gc.young_gc_count", "op": ">", "threshold": 5},
        ],
        apply_template="Restart JVM with -XX:NewSize={value} -XX:MaxNewSize={value}",
        verify_template="jcmd {pid} GC.heap_info",
        rollback_template="Restart JVM without NewSize override",
        config_path="jvm.young_gen_size",
    ),
    TuningRule(
        parameter_name="jvm.metaspace_size",
        display_name="JVM Metaspace Size",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.EASY,
        risk=TuningRiskLevel.LOW,
        description="Metaspace holds class metadata. Too small triggers Full GC on class loading bursts.",
        recommended_values={
            "jvm_general": "256m",
            "jvm_jit_intensive": "512m",
            "bigdata_batch": "512m",
            "_default": "256m",
        },
        base_impact={
            "jvm_general": 0.2, "jvm_jit_intensive": 0.4,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "jvm.gc.metaspace_used_mb", "op": ">", "threshold": 200, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "jvm.gc.metaspace_used_mb", "op": ">", "threshold": 100},
        ],
        apply_template="Restart JVM with -XX:MetaspaceSize={value} -XX:MaxMetaspaceSize={value}",
        verify_template="jcmd {pid} VM.flags | grep MetaspaceSize",
        rollback_template="Restart JVM without MetaspaceSize override",
        config_path="jvm.metaspace_size",
    ),
    TuningRule(
        parameter_name="jvm.jit_compiler_threads",
        display_name="JVM JIT Compiler Threads",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.EASY,
        risk=TuningRiskLevel.LOW,
        description="Number of C1/C2 compiler threads. More threads speed up warmup but consume CPU.",
        recommended_values={
            "jvm_general": "c1:cpu/4,c2:cpu/4",
            "jvm_jit_intensive": "c1:cpu/2,c2:cpu/2",
            "_default": "c1:cpu/4,c2:cpu/4",
        },
        base_impact={
            "jvm_general": 0.1, "jvm_jit_intensive": 0.5,
            "_default": 0.1,
        },
        impact_conditions=[
            {"metric": "jvm.jit.compilations_per_sec", "op": ">", "threshold": 5, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "jvm.jit.compilations_per_sec", "op": ">", "threshold": 2},
        ],
        apply_template="Restart JVM with -XX:CICompilerCount={value}",
        verify_template="jcmd {pid} VM.flags | grep CICompilerCount",
        rollback_template="Restart JVM without CICompilerCount override",
        config_path="jvm.jit_compiler_threads",
    ),
    TuningRule(
        parameter_name="jvm.thread_stack_size",
        display_name="JVM Thread Stack Size",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.MODERATE,
        risk=TuningRiskLevel.MEDIUM,
        description="Per-thread stack size. Smaller stacks save memory with many threads but risk StackOverflow.",
        recommended_values={
            "jvm_general": "512k",
            "microservice": "256k",
            "bigdata_batch": "1024k",
            "_default": "512k",
        },
        base_impact={
            "jvm_general": 0.1, "microservice": 0.3, "bigdata_batch": 0.1,
            "_default": 0.05,
        },
        impact_conditions=[
            {"metric": "jvm.threads.total_threads", "op": ">", "threshold": 500, "boost": 0.15},
        ],
        prerequisites=[
            {"metric": "jvm.threads.total_threads", "op": ">", "threshold": 200},
        ],
        apply_template="Restart JVM with -Xss{value}",
        verify_template="jcmd {pid} VM.flags | grep ThreadStackSize",
        rollback_template="Restart JVM without -Xss override",
        config_path="jvm.thread_stack_size",
    ),
    TuningRule(
        parameter_name="jvm.large_pages",
        display_name="JVM Large Pages",
        layer=TuningLayer.JVM,
        difficulty=TuningDifficulty.HARD,
        risk=TuningRiskLevel.MEDIUM,
        description="Use large pages for Java heap to reduce TLB misses. Requires OS hugepage pre-allocation.",
        recommended_values={
            "jvm_general": "true",
            "jvm_gc_heavy": "true",
            "database_oltp": "true",
            "memory_bound": "true",
            "_default": "false",
        },
        base_impact={
            "jvm_general": 0.2, "jvm_gc_heavy": 0.3, "database_oltp": 0.3,
            "memory_bound": 0.3, "_default": 0.05,
        },
        impact_conditions=[
            {"metric": "memory.tlb_mpki", "op": ">", "threshold": 1.0, "boost": 0.2},
        ],
        prerequisites=[
            {"metric": "memory.tlb_mpki", "op": ">", "threshold": 0.5},
        ],
        apply_template="Restart JVM with -XX:+UseLargePages",
        verify_template="jcmd {pid} VM.flags | grep UseLargePages",
        rollback_template="Restart JVM with -XX:-UseLargePages",
        config_path="jvm.large_pages",
    ),
]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class OptimizationAnalyzer:
    """Analyze platform configuration gaps and generate optimization recommendations."""

    def analyze(
        self,
        fv: WorkloadFeatureVector,
    ) -> OptimizationReport:
        """Analyze one workload scenario against its current platform config."""
        config = fv.platform_config
        scenario_type = fv.scenario_type.value

        recommendations: List[OptimizationRecommendation] = []
        for rule in TUNING_RULES:
            rec = self._evaluate_rule(rule, fv, config, scenario_type)
            if rec is not None:
                recommendations.append(rec)

        # Sort by priority descending
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        # Build layer summaries
        layers: Dict[str, LayerSummary] = {}
        for layer in TuningLayer:
            layer_recs = [r for r in recommendations if r.layer == layer]
            gaps = sum(1 for r in layer_recs if r.gap_detected is True)
            max_impact = max((r.impact_score for r in layer_recs), default=0.0)
            layers[layer.value] = LayerSummary(
                layer=layer,
                total_parameters=len(layer_recs),
                gaps_found=gaps,
                max_impact=max_impact,
                recommendations=layer_recs,
            )

        opt_score = self._compute_optimization_score(recommendations)

        total_gaps = sum(ls.gaps_found for ls in layers.values())
        summary = (
            f"Optimization score: {opt_score:.0f}/100 — "
            f"{total_gaps} configuration gaps detected across "
            f"{len(recommendations)} parameters."
        )

        return OptimizationReport(
            scenario_name=fv.scenario_name,
            scenario_type=fv.scenario_type,
            layers=layers,
            all_recommendations=recommendations,
            optimization_score=opt_score,
            summary=summary,
        )

    def cross_scenario_analysis(
        self,
        feature_vectors: List[WorkloadFeatureVector],
    ) -> CrossScenarioOptimizationReport:
        """Analyze across multiple scenarios."""
        per_scenario: Dict[str, OptimizationReport] = {}
        for fv in feature_vectors:
            per_scenario[fv.scenario_name] = self.analyze(fv)

        # Build benefit matrix
        param_names = [r.parameter_name for r in TUNING_RULES]
        matrix_data = {}
        for name, report in per_scenario.items():
            row = {}
            for rec in report.all_recommendations:
                row[rec.parameter_name] = rec.impact_score
            matrix_data[name] = row
        matrix = pd.DataFrame(matrix_data).T
        matrix = matrix.reindex(columns=param_names)

        # Find universal recommendations (same recommended value for all scenarios)
        universal: List[OptimizationRecommendation] = []
        conflicting: List[Dict[str, Any]] = []

        for rule in TUNING_RULES:
            rec_values = {}
            recs_for_param: List[OptimizationRecommendation] = []
            for name, report in per_scenario.items():
                for rec in report.all_recommendations:
                    if rec.parameter_name == rule.parameter_name:
                        rec_values[name] = rec.recommended_value
                        recs_for_param.append(rec)
                        break

            unique_values = set(rec_values.values())
            if len(unique_values) == 1 and recs_for_param:
                # All agree — pick the one with highest impact
                best = max(recs_for_param, key=lambda r: r.impact_score)
                universal.append(best)
            elif len(unique_values) > 1:
                conflicting.append({
                    "parameter": rule.display_name,
                    "parameter_name": rule.parameter_name,
                    "scenarios_disagree": {
                        k: v for k, v in rec_values.items()
                    },
                })

        universal.sort(key=lambda r: r.priority_score, reverse=True)

        return CrossScenarioOptimizationReport(
            per_scenario=per_scenario,
            universal_recommendations=universal,
            conflicting_parameters=conflicting,
            parameter_benefit_matrix=matrix,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _evaluate_rule(
        self,
        rule: TuningRule,
        fv: WorkloadFeatureVector,
        config: Optional[PlatformConfigSnapshot],
        scenario_type: str,
    ) -> Optional[OptimizationRecommendation]:
        """Evaluate a single tuning rule against a workload + config."""
        # 1. Recommended value
        recommended = rule.recommended_values.get(
            scenario_type, rule.recommended_values.get("_default", ""))

        # 2. Current value
        current = self._get_current_value(rule, config)

        # 3. Impact score
        base = rule.base_impact.get(
            scenario_type, rule.base_impact.get("_default", 0.2))

        # Skip rules explicitly set to 0 impact for this scenario type
        if base <= 0.0 and scenario_type in rule.base_impact:
            return None

        # Skip rules whose prerequisites are not met
        if not self._check_prerequisites(rule, fv):
            return None

        boost = 0.0
        reasoning_parts = []
        for cond in rule.impact_conditions:
            metric_val = self._resolve_metric(fv, cond["metric"])
            if metric_val is not None:
                op = cond["op"]
                threshold = cond["threshold"]
                matched = False
                if op == ">" and metric_val > threshold:
                    matched = True
                elif op == "<" and metric_val < threshold:
                    matched = True
                elif op == ">=" and metric_val >= threshold:
                    matched = True
                elif op == "<=" and metric_val <= threshold:
                    matched = True
                if matched:
                    boost += cond["boost"]
                    reasoning_parts.append(
                        f"{cond['metric']}={metric_val:.2f} (threshold {op} {threshold})")
        impact = clamp(base + boost)

        # 4. Gap detection
        gap = None
        if current != "unknown":
            gap = not self._values_match(current, recommended, rule)

        # 5. Priority
        mult = DIFFICULTY_MULTIPLIER.get(rule.difficulty, 0.5)
        priority = impact * mult

        # 6. Reasoning
        if reasoning_parts:
            reasoning = f"Base impact={base:.2f} for {scenario_type}. " + \
                        "Feature boosts: " + "; ".join(reasoning_parts)
        else:
            reasoning = f"Base impact={base:.2f} for {scenario_type} scenario."

        # 7. Commands
        apply_cmds = [rule.apply_template.format(
            value=recommended, current=current)] if rule.apply_template else []
        verify_cmds = [rule.verify_template] if rule.verify_template else []
        rollback_cmds = [rule.rollback_template.format(
            value=recommended, current=current)] if rule.rollback_template else []

        return OptimizationRecommendation(
            parameter_name=rule.parameter_name,
            display_name=rule.display_name,
            layer=rule.layer,
            difficulty=rule.difficulty,
            risk=rule.risk,
            description=rule.description,
            current_value=current,
            recommended_value=recommended,
            impact_score=round(impact, 3),
            priority_score=round(priority, 3),
            gap_detected=gap,
            reasoning=reasoning,
            apply_commands=apply_cmds,
            verify_commands=verify_cmds,
            rollback_commands=rollback_cmds,
        )

    def _get_current_value(
        self, rule: TuningRule, config: Optional[PlatformConfigSnapshot]
    ) -> str:
        """Extract the current value for a tuning parameter from the config snapshot."""
        if config is None or not rule.config_path:
            return "unknown"

        parts = rule.config_path.split(".")
        obj: Any = config
        try:
            for part in parts:
                if isinstance(obj, dict):
                    obj = obj.get(part)
                else:
                    obj = getattr(obj, part, None)
                if obj is None:
                    return "unknown"
        except Exception:
            return "unknown"

        # Handle special cases
        if isinstance(obj, dict):
            # e.g., io_schedulers dict — return first value or summary
            if obj:
                return str(list(obj.values())[0])
            return "unknown"
        if isinstance(obj, bool):
            return str(obj)
        return str(obj)

    def _resolve_metric(
        self, fv: WorkloadFeatureVector, dotpath: str
    ) -> Optional[float]:
        """Resolve a dot-path like 'memory.tlb_mpki' to a numeric value."""
        parts = dotpath.split(".")
        obj: Any = fv
        try:
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
            return float(obj) if obj is not None else None
        except (TypeError, ValueError):
            return None

    def _check_prerequisites(
        self, rule: TuningRule, fv: WorkloadFeatureVector
    ) -> bool:
        """Check if all prerequisites for a rule are met.

        A rule with no prerequisites always passes.
        A rule with prerequisites is suppressed unless every condition is met.
        """
        if not rule.prerequisites:
            return True
        for cond in rule.prerequisites:
            metric_val = self._resolve_metric(fv, cond["metric"])
            if metric_val is None:
                return False
            op = cond["op"]
            threshold = cond["threshold"]
            if op == ">" and not (metric_val > threshold):
                return False
            elif op == "<" and not (metric_val < threshold):
                return False
            elif op == ">=" and not (metric_val >= threshold):
                return False
            elif op == "<=" and not (metric_val <= threshold):
                return False
        return True

    @staticmethod
    def _values_match(
        current: str, recommended: str, rule: TuningRule
    ) -> bool:
        """Check if current value matches recommended (with tolerance)."""
        c = current.strip().lower()
        r = recommended.strip().lower()

        # Direct match
        if c == r:
            return True

        # Bool match: "True"/"1"/"on" are equivalent
        bool_true = {"true", "1", "on", "yes", "active"}
        bool_false = {"false", "0", "off", "no", "inactive"}
        if c in bool_true and r in bool_true:
            return True
        if c in bool_false and r in bool_false:
            return True

        # Special: "auto" or "max" or "default" are considered non-gap
        # if we can't compare numerically
        if r in ("auto", "max", "default"):
            return False  # Can't verify, flag as gap for review

        # Numeric comparison with 10% tolerance
        try:
            cv = float(c)
            rv = float(r)
            if rv == 0:
                return cv == 0
            return abs(cv - rv) / max(abs(rv), 1) < 0.1
        except ValueError:
            pass

        return False

    @staticmethod
    def _compute_optimization_score(
        recommendations: List[OptimizationRecommendation],
    ) -> float:
        """Compute 0-100 score representing how well-tuned the platform is."""
        if not recommendations:
            return 100.0

        total_weight = 0.0
        matched_weight = 0.0
        for rec in recommendations:
            w = rec.impact_score
            total_weight += w
            if rec.gap_detected is False:
                matched_weight += w
            elif rec.gap_detected is None:
                # Unknown — give partial credit
                matched_weight += w * 0.5

        if total_weight <= 0:
            return 100.0
        return round((matched_weight / total_weight) * 100, 1)
