"""Hardware design parameter sensitivity analysis.

Maps workload characteristics to hardware design parameter sensitivities.
This is the core deliverable for chip architects — it answers the question:
"Which hardware knobs matter most for each workload?"

Key output: cross_workload_matrix() generates a scenarios x parameters
sensitivity matrix that enables trade-off analysis across workloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..model.enums import AccessPattern
from ..model.schema import WorkloadFeatureVector
from ..utils.units import clamp

log = logging.getLogger(__name__)


@dataclass
class DesignParameter:
    """A hardware design parameter under exploration."""
    name: str
    display_name: str
    unit: str
    current_value: str  # Current generation baseline
    range_values: List[str]
    area_cost: str  # Qualitative: "low", "medium", "high"


# Hardware design parameters under exploration
DESIGN_PARAMETERS: Dict[str, DesignParameter] = {
    "l1d_cache_size": DesignParameter(
        "l1d_cache_size", "L1D Cache Size", "KB", "64", ["32", "48", "64", "128"], "medium"),
    "l1i_cache_size": DesignParameter(
        "l1i_cache_size", "L1I Cache Size", "KB", "64", ["32", "48", "64", "128"], "medium"),
    "l2_cache_size": DesignParameter(
        "l2_cache_size", "L2 Cache Size", "KB", "512", ["256", "512", "1024", "2048"], "medium"),
    "l3_cache_size": DesignParameter(
        "l3_cache_size", "L3 Cache Size", "MB", "32", ["16", "32", "48", "64", "96", "128"], "high"),
    "l3_associativity": DesignParameter(
        "l3_associativity", "L3 Associativity", "ways", "16", ["8", "16", "32"], "low"),
    "memory_bandwidth": DesignParameter(
        "memory_bandwidth", "Memory Bandwidth", "GB/s/ch", "25.6",
        ["25.6", "38.4", "51.2"], "high"),
    "memory_channels": DesignParameter(
        "memory_channels", "Memory Channels", "ch", "8", ["4", "6", "8", "12"], "high"),
    "dispatch_width": DesignParameter(
        "dispatch_width", "Dispatch Width", "uops/cyc", "4", ["4", "6", "8", "10"], "high"),
    "rob_entries": DesignParameter(
        "rob_entries", "ROB Entries", "entries", "192", ["128", "192", "256", "384", "512"], "medium"),
    "btb_entries": DesignParameter(
        "btb_entries", "BTB Entries", "entries", "4096",
        ["4096", "8192", "16384"], "low"),
    "prefetch_aggressiveness": DesignParameter(
        "prefetch_aggressiveness", "Prefetch Aggressiveness", "level", "moderate",
        ["conservative", "moderate", "aggressive"], "low"),
    "core_count": DesignParameter(
        "core_count", "Core Count", "cores", "64", ["32", "48", "64", "96", "128"], "high"),
    "simd_width": DesignParameter(
        "simd_width", "SIMD Width", "bits", "128", ["128", "256", "512"], "high"),
}


@dataclass
class SensitivityScore:
    """Sensitivity score for one (workload, parameter) pair."""
    parameter: str
    score: float  # 0.0 (insensitive) to 1.0 (highly sensitive)
    reasoning: str
    confidence: str  # "high", "medium", "low"


@dataclass
class DesignRecommendation:
    """A ranked design recommendation."""
    parameter: str
    direction: str  # "increase", "decrease", "optimize"
    priority: float  # weighted score
    justification: str
    affected_workloads: List[str]
    area_cost: str


@dataclass
class DesignSensitivityReport:
    """Complete sensitivity analysis across workloads."""
    per_workload: Dict[str, List[SensitivityScore]]  # scenario -> scores
    matrix: Optional[pd.DataFrame] = None  # scenarios x parameters
    recommendations: List[DesignRecommendation] = field(default_factory=list)


class DesignSpaceExplorer:
    """Map workload characteristics to hardware design parameter sensitivities."""

    def compute_sensitivity(
        self, fv: WorkloadFeatureVector
    ) -> List[SensitivityScore]:
        """Compute sensitivity scores for all design parameters."""
        scores = [
            self._l1d_cache_sensitivity(fv),
            self._l1i_cache_sensitivity(fv),
            self._l2_cache_sensitivity(fv),
            self._l3_cache_sensitivity(fv),
            self._l3_associativity_sensitivity(fv),
            self._memory_bw_sensitivity(fv),
            self._memory_channels_sensitivity(fv),
            self._dispatch_width_sensitivity(fv),
            self._rob_sensitivity(fv),
            self._btb_sensitivity(fv),
            self._prefetch_sensitivity(fv),
            self._core_count_sensitivity(fv),
            self._simd_width_sensitivity(fv),
        ]
        return scores

    def cross_workload_matrix(
        self, feature_vectors: List[WorkloadFeatureVector]
    ) -> pd.DataFrame:
        """Generate a workloads x design_parameters sensitivity matrix.

        Each cell = sensitivity score [0, 1].
        This is the key deliverable for architects — enables trade-off
        analysis like "which parameter benefits the most workloads?"
        """
        data = {}
        for fv in feature_vectors:
            scores = self.compute_sensitivity(fv)
            data[fv.scenario_name] = {s.parameter: s.score for s in scores}

        df = pd.DataFrame(data).T
        df.columns.name = "Design Parameter"
        df.index.name = "Workload"
        return df

    def recommend_design_tradeoffs(
        self,
        feature_vectors: List[WorkloadFeatureVector],
        priority_weights: Optional[Dict[str, float]] = None,
    ) -> List[DesignRecommendation]:
        """Generate ranked design recommendations.

        Args:
            feature_vectors: All workload profiles
            priority_weights: Optional weights per scenario type
                (e.g., {"database_oltp": 2.0} to prioritize database workloads)
        """
        matrix = self.cross_workload_matrix(feature_vectors)

        # Apply priority weights
        if priority_weights:
            for idx in matrix.index:
                fv = next(f for f in feature_vectors if f.scenario_name == idx)
                weight = priority_weights.get(fv.scenario_type.value, 1.0)
                matrix.loc[idx] *= weight

        recommendations = []
        for param in matrix.columns:
            scores = matrix[param]
            avg_score = scores.mean()
            max_score = scores.max()
            affected = scores[scores > 0.5].index.tolist()

            if avg_score < 0.2 and max_score < 0.5:
                continue  # Not worth recommending

            param_info = DESIGN_PARAMETERS.get(param)
            area_cost = param_info.area_cost if param_info else "unknown"

            # Compute cost-effectiveness: benefit / area_cost
            cost_multiplier = {"low": 1.0, "medium": 0.7, "high": 0.4}.get(area_cost, 0.5)
            priority = avg_score * cost_multiplier

            recommendations.append(DesignRecommendation(
                parameter=param,
                direction="increase",
                priority=round(priority, 3),
                justification=(
                    f"Average sensitivity {avg_score:.0%} across {len(feature_vectors)} "
                    f"workloads, peak {max_score:.0%}. "
                    f"Benefits {len(affected)} workloads significantly (>50% sensitivity). "
                    f"Area cost: {area_cost}."
                ),
                affected_workloads=affected,
                area_cost=area_cost,
            ))

        # Sort by priority (cost-effective benefit)
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations

    def full_analysis(
        self,
        feature_vectors: List[WorkloadFeatureVector],
        priority_weights: Optional[Dict[str, float]] = None,
    ) -> DesignSensitivityReport:
        """Run complete sensitivity analysis."""
        per_workload = {}
        for fv in feature_vectors:
            per_workload[fv.scenario_name] = self.compute_sensitivity(fv)

        matrix = self.cross_workload_matrix(feature_vectors)
        recs = self.recommend_design_tradeoffs(feature_vectors, priority_weights)

        return DesignSensitivityReport(
            per_workload=per_workload,
            matrix=matrix,
            recommendations=recs,
        )

    # -----------------------------------------------------------------------
    # Individual parameter sensitivity scoring functions
    # -----------------------------------------------------------------------

    def _l1d_cache_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """L1D cache size sensitivity.

        High if: L1D MPKI is high but L2 miss rate is low (data fits in L2 but not L1D).
        """
        mpki = fv.cache.l1d_mpki
        # Sigmoid-like scoring: 0 at MPKI<5, ~0.5 at MPKI=20, ~1.0 at MPKI>50
        score = clamp(1.0 - 1.0 / (1.0 + mpki / 20.0))

        # Boost if good spatial locality (cache would be effective)
        if fv.cache.spatial_locality_score and fv.cache.spatial_locality_score > 0.6:
            score = min(1.0, score * 1.2)

        return SensitivityScore(
            parameter="l1d_cache_size",
            score=round(score, 3),
            reasoning=f"L1D MPKI={mpki:.1f}, spatial locality={fv.cache.spatial_locality_score}",
            confidence="high",
        )

    def _l1i_cache_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """L1I cache size sensitivity. High if frontend-bound with L1I misses."""
        mpki = fv.cache.l1i_mpki
        fe = fv.compute.topdown_l1.frontend_bound

        score = clamp(1.0 - 1.0 / (1.0 + mpki / 5.0))
        # Boost by frontend_bound fraction
        score = score * (0.5 + fe)

        return SensitivityScore(
            parameter="l1i_cache_size",
            score=round(clamp(score), 3),
            reasoning=f"L1I MPKI={mpki:.1f}, frontend_bound={fe:.0%}",
            confidence="high" if fe > 0.2 else "medium",
        )

    def _l2_cache_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """L2 cache sensitivity. High if L2 MPKI high but L3 hit rate is good."""
        mpki = fv.cache.l2_mpki
        score = clamp(1.0 - 1.0 / (1.0 + mpki / 15.0))

        return SensitivityScore(
            parameter="l2_cache_size",
            score=round(score, 3),
            reasoning=f"L2 MPKI={mpki:.1f}, L2 miss rate={fv.cache.l2_miss_rate:.1%}",
            confidence="high",
        )

    def _l3_cache_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """L3 cache size sensitivity.

        High if: L3 MPKI > 5, working set could benefit from more cache,
                 good locality (cache-friendly access pattern).
        Low if:  L3 MPKI < 1 (fits in L2), or random access (no cache reuse).
        """
        mpki = fv.cache.l3_mpki
        score = clamp(1.0 - 1.0 / (1.0 + mpki / 5.0))

        # Penalize if random access pattern (larger cache won't help much)
        if fv.memory.access_pattern == AccessPattern.RANDOM:
            score *= 0.5
        elif fv.cache.spatial_locality_score and fv.cache.spatial_locality_score < 0.3:
            score *= 0.6

        return SensitivityScore(
            parameter="l3_cache_size",
            score=round(clamp(score), 3),
            reasoning=(
                f"L3 MPKI={mpki:.1f}, miss rate={fv.cache.l3_miss_rate:.1%}, "
                f"spatial locality={fv.cache.spatial_locality_score}"
            ),
            confidence="high",
        )

    def _l3_associativity_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """L3 associativity. Matters if L3 miss rate is high but MPKI moderate."""
        # High miss rate with moderate MPKI suggests conflict misses
        miss_rate = fv.cache.l3_miss_rate
        mpki = fv.cache.l3_mpki

        # Conflict miss indicator: miss rate high but MPKI not extreme
        if miss_rate > 0.3 and mpki < 10:
            score = 0.6
        elif miss_rate > 0.2:
            score = 0.4
        else:
            score = 0.1

        return SensitivityScore(
            parameter="l3_associativity",
            score=round(score, 3),
            reasoning=f"L3 miss rate={miss_rate:.1%} with MPKI={mpki:.1f}",
            confidence="low",  # Hard to separate conflict from capacity misses
        )

    def _memory_bw_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """Memory bandwidth sensitivity.

        High if: bandwidth utilization > 50%, streaming patterns.
        """
        bw_util = fv.memory.bandwidth_utilization
        be_mem = fv.compute.topdown_l1.backend_bound

        score = clamp(bw_util * 1.5)  # Linear with boost
        # Reinforce with backend-bound
        score = max(score, be_mem * bw_util * 2)

        return SensitivityScore(
            parameter="memory_bandwidth",
            score=round(clamp(score), 3),
            reasoning=f"BW utilization={bw_util:.0%}, backend_bound={be_mem:.0%}",
            confidence="high" if bw_util > 0.3 else "medium",
        )

    def _memory_channels_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """Memory channel count. Correlated with bandwidth sensitivity."""
        bw_util = fv.memory.bandwidth_utilization
        score = clamp(bw_util * 1.2)

        return SensitivityScore(
            parameter="memory_channels",
            score=round(clamp(score), 3),
            reasoning=f"BW utilization={bw_util:.0%}",
            confidence="medium",
        )

    def _dispatch_width_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """Dispatch width sensitivity.

        High if retiring fraction is high (ILP-rich, extracting parallelism)
        and IPC is close to current dispatch width.
        """
        retiring = fv.compute.topdown_l1.retiring
        ipc_ratio = fv.compute.ipc / 4  # Assuming 4-wide baseline

        # High retiring + high IPC = benefits from wider pipeline
        score = retiring * ipc_ratio

        return SensitivityScore(
            parameter="dispatch_width",
            score=round(clamp(score), 3),
            reasoning=f"Retiring={retiring:.0%}, IPC={fv.compute.ipc:.2f} (ratio={ipc_ratio:.0%})",
            confidence="medium",
        )

    def _rob_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """ROB (Reorder Buffer) depth sensitivity.

        Benefits workloads with long memory latency that need more OoO window.
        """
        be = fv.compute.topdown_l1.backend_bound
        l3_mpki = fv.cache.l3_mpki

        # Backend-memory bound with high L3 MPKI = needs larger OoO window
        score = be * clamp(l3_mpki / 10.0)

        return SensitivityScore(
            parameter="rob_entries",
            score=round(clamp(score), 3),
            reasoning=f"Backend bound={be:.0%}, L3 MPKI={l3_mpki:.1f}",
            confidence="medium",
        )

    def _btb_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """BTB (Branch Target Buffer) sensitivity.

        High if branch MPKI > 5 or high indirect branch ratio.
        """
        br_mpki = fv.branch.branch_mpki
        indirect = fv.branch.indirect_branch_ratio or 0.0

        score = clamp(br_mpki / 10.0)
        if indirect > 0.3:
            score = max(score, 0.7)
        elif indirect > 0.15:
            score = max(score, 0.5)

        return SensitivityScore(
            parameter="btb_entries",
            score=round(clamp(score), 3),
            reasoning=f"Branch MPKI={br_mpki:.1f}, indirect ratio={indirect:.0%}",
            confidence="high" if br_mpki > 5 else "medium",
        )

    def _prefetch_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """Prefetcher aggressiveness sensitivity.

        High for streaming workloads with good spatial locality.
        Low for random access patterns.
        """
        spatial = fv.cache.spatial_locality_score or 0.5
        l2_mpki = fv.cache.l2_mpki

        # Streaming + high L2 MPKI = prefetcher would help
        score = spatial * clamp(l2_mpki / 15.0)

        return SensitivityScore(
            parameter="prefetch_aggressiveness",
            score=round(clamp(score), 3),
            reasoning=(
                f"Spatial locality={spatial:.2f}, L2 MPKI={l2_mpki:.1f}, "
                f"access pattern={fv.memory.access_pattern}"
            ),
            confidence="medium",
        )

    def _core_count_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """Core count sensitivity based on scalability profile."""
        if fv.scalability is None:
            # Estimate from concurrency profile
            thread_count = fv.concurrency.thread_count
            lock_pct = fv.concurrency.lock_contention_pct or 0
            if thread_count > 32 and lock_pct < 10:
                score = 0.7
            elif thread_count > 8:
                score = 0.5
            else:
                score = 0.3
        else:
            serial = fv.scalability.amdahl_serial_fraction or 0.1
            # Low serial fraction = scales well = benefits from more cores
            score = 1.0 - serial

        return SensitivityScore(
            parameter="core_count",
            score=round(clamp(score), 3),
            reasoning=f"Threads={fv.concurrency.thread_count}, serial fraction estimated",
            confidence="low" if fv.scalability is None else "high",
        )

    def _simd_width_sensitivity(self, fv: WorkloadFeatureVector) -> SensitivityScore:
        """SIMD width (NEON 128-bit vs SVE 256/512-bit) sensitivity.

        High if workload has significant FP/vector instruction mix.
        """
        simd = fv.compute.simd_utilization
        vec_ratio = fv.compute.instruction_mix.vector_ratio
        fp_ratio = fv.compute.instruction_mix.fp_ratio

        score = max(simd, vec_ratio + fp_ratio * 0.5)

        return SensitivityScore(
            parameter="simd_width",
            score=round(clamp(score), 3),
            reasoning=(
                f"SIMD util={simd:.0%}, vector ratio={vec_ratio:.0%}, "
                f"FP ratio={fp_ratio:.0%}"
            ),
            confidence="high" if simd > 0.1 else "low",
        )
