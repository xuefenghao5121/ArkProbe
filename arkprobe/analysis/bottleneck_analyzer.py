"""TopDown bottleneck analysis for ARM Kunpeng processors.

Identifies micro-architectural bottlenecks using the ARM TopDown methodology
and generates architect-readable analysis with design parameter recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..model.enums import BottleneckCategory
from ..model.schema import WorkloadFeatureVector

log = logging.getLogger(__name__)


@dataclass
class BottleneckDetail:
    """Detail analysis for a specific bottleneck component."""
    category: str
    score: float  # 0-1 fraction
    indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BottleneckReport:
    """Complete bottleneck analysis report for one workload."""
    scenario_name: str
    primary_bottleneck: BottleneckCategory
    primary_score: float
    topdown_l1: dict  # {frontend_bound, backend_bound, retiring, bad_speculation}
    details: List[BottleneckDetail] = field(default_factory=list)
    architect_notes: List[str] = field(default_factory=list)
    summary: str = ""


class BottleneckAnalyzer:
    """Identify micro-architectural bottlenecks using TopDown methodology."""

    # Thresholds for bottleneck classification
    FRONTEND_THRESHOLD = 0.20
    BACKEND_THRESHOLD = 0.30
    BAD_SPEC_THRESHOLD = 0.15
    RETIRING_GOOD_THRESHOLD = 0.50

    # Sub-thresholds for indicators
    HIGH_L3_MPKI = 5.0
    HIGH_BRANCH_MPKI = 5.0
    HIGH_L1I_MPKI = 3.0
    HIGH_TLB_MPKI = 1.0
    HIGH_BW_UTIL = 0.5
    HIGH_INDIRECT_RATIO = 0.3

    def __init__(self, dispatch_width: int = 4):
        self.dispatch_width = dispatch_width

    def analyze(self, fv: WorkloadFeatureVector) -> BottleneckReport:
        """Produce a structured bottleneck report from the feature vector."""
        td = fv.compute.topdown_l1
        topdown_dict = {
            "frontend_bound": td.frontend_bound,
            "backend_bound": td.backend_bound,
            "retiring": td.retiring,
            "bad_speculation": td.bad_speculation,
        }

        primary = self._classify_primary(td.frontend_bound, td.backend_bound,
                                         td.retiring, td.bad_speculation)

        details = []
        details.extend(self._analyze_frontend(fv))
        details.extend(self._analyze_backend(fv))
        details.extend(self._analyze_bad_speculation(fv))

        notes = self._generate_architect_notes(fv, primary, details)

        report = BottleneckReport(
            scenario_name=fv.scenario_name,
            primary_bottleneck=primary,
            primary_score=self._primary_score(td, primary),
            topdown_l1=topdown_dict,
            details=details,
            architect_notes=notes,
        )
        report.summary = self._generate_summary(report)

        return report

    def _classify_primary(
        self, fe: float, be: float, ret: float, bad: float
    ) -> BottleneckCategory:
        """Determine the primary bottleneck category."""
        if ret >= self.RETIRING_GOOD_THRESHOLD:
            return BottleneckCategory.WELL_BALANCED

        # Find the dominant non-retiring component
        components = {
            BottleneckCategory.FRONTEND_BOUND: fe,
            BottleneckCategory.BAD_SPECULATION: bad,
        }
        # Backend is split into memory and core
        if be > fe and be > bad:
            # Use proxy: if L3 MPKI is high or bandwidth is high, it's memory-bound
            return BottleneckCategory.BACKEND_MEMORY_BOUND

        dominant = max(components, key=components.get)
        if components[dominant] < 0.15:
            return BottleneckCategory.WELL_BALANCED

        return dominant

    def _primary_score(self, td, category: BottleneckCategory) -> float:
        mapping = {
            BottleneckCategory.FRONTEND_BOUND: td.frontend_bound,
            BottleneckCategory.BACKEND_MEMORY_BOUND: td.backend_bound,
            BottleneckCategory.BACKEND_CORE_BOUND: td.backend_bound,
            BottleneckCategory.BAD_SPECULATION: td.bad_speculation,
            BottleneckCategory.WELL_BALANCED: td.retiring,
        }
        return mapping.get(category, 0.0)

    def _analyze_frontend(self, fv: WorkloadFeatureVector) -> List[BottleneckDetail]:
        """Analyze frontend bottleneck indicators."""
        details = []
        fe = fv.compute.topdown_l1.frontend_bound

        if fe < self.FRONTEND_THRESHOLD:
            return details

        indicators = []
        recommendations = []

        # I-cache pressure
        if fv.cache.l1i_mpki > self.HIGH_L1I_MPKI:
            indicators.append(
                f"L1I MPKI = {fv.cache.l1i_mpki:.1f} (high) — instruction cache misses "
                f"are stalling the frontend"
            )
            recommendations.append(
                "Consider larger L1I cache or instruction prefetching improvements"
            )

        # iTLB pressure
        if fv.memory.tlb_mpki and fv.memory.tlb_mpki > self.HIGH_TLB_MPKI:
            indicators.append(
                f"TLB MPKI = {fv.memory.tlb_mpki:.1f} — large code footprint "
                f"causing frequent page walks"
            )
            recommendations.append(
                "Consider larger iTLB, more TLB levels, or HugePage support for code"
            )

        # High branch density
        if fv.branch.branch_density and fv.branch.branch_density > 0.2:
            indicators.append(
                f"Branch density = {fv.branch.branch_density:.3f} — one in "
                f"{int(1/fv.branch.branch_density)} instructions is a branch"
            )
            recommendations.append(
                "Consider macro-op fusion for compare+branch patterns"
            )

        if not indicators:
            indicators.append("Frontend bound but specific cause unclear from available metrics")

        details.append(BottleneckDetail(
            category="Frontend Bound",
            score=fe,
            indicators=indicators,
            recommendations=recommendations,
        ))

        return details

    def _analyze_backend(self, fv: WorkloadFeatureVector) -> List[BottleneckDetail]:
        """Analyze backend bottleneck: split into memory-bound and core-bound."""
        details = []
        be = fv.compute.topdown_l1.backend_bound

        if be < self.BACKEND_THRESHOLD * 0.5:
            return details

        # Memory-bound indicators
        mem_indicators = []
        mem_recs = []

        if fv.cache.l3_mpki > self.HIGH_L3_MPKI:
            mem_indicators.append(
                f"L3 MPKI = {fv.cache.l3_mpki:.1f} — significant last-level cache "
                f"pressure driving memory accesses"
            )
            mem_recs.append(
                f"Increasing L3 cache size could reduce backend stalls. "
                f"Current L3 miss rate = {fv.cache.l3_miss_rate:.1%}"
            )

        if fv.cache.l2_mpki > 10:
            mem_indicators.append(
                f"L2 MPKI = {fv.cache.l2_mpki:.1f} — working set exceeds L1D capacity"
            )
            mem_recs.append("Consider larger L2 cache or improved L2 prefetching")

        if fv.memory.bandwidth_utilization > self.HIGH_BW_UTIL:
            mem_indicators.append(
                f"Memory bandwidth utilization = {fv.memory.bandwidth_utilization:.0%} "
                f"— approaching memory controller saturation"
            )
            mem_recs.append(
                "More memory channels or higher-frequency DIMMs needed"
            )

        if fv.memory.numa_local_ratio is not None and fv.memory.numa_local_ratio < 0.8:
            mem_indicators.append(
                f"NUMA local access ratio = {fv.memory.numa_local_ratio:.0%} "
                f"— significant remote memory traffic"
            )
            mem_recs.append(
                "Interconnect bandwidth and NUMA topology improvements"
            )

        if mem_indicators:
            details.append(BottleneckDetail(
                category="Backend: Memory Bound",
                score=be,
                indicators=mem_indicators,
                recommendations=mem_recs,
            ))

        # Core-bound indicators
        core_indicators = []
        core_recs = []

        ipc_ratio = fv.compute.ipc / max(1, self.dispatch_width)  # dispatch width from hardware config
        if ipc_ratio > 0.7 and not mem_indicators:
            core_indicators.append(
                f"IPC = {fv.compute.ipc:.2f} approaching dispatch width — "
                f"execution unit saturation"
            )
            core_recs.append("Wider dispatch or more execution units")

        if fv.compute.instruction_mix.fp_ratio > 0.2:
            core_indicators.append(
                f"FP instruction ratio = {fv.compute.instruction_mix.fp_ratio:.0%} "
                f"— FP execution unit may be bottleneck"
            )
            core_recs.append("Additional FP/SIMD execution units")

        if core_indicators:
            details.append(BottleneckDetail(
                category="Backend: Core Bound",
                score=be * (1 - fv.memory.bandwidth_utilization),
                indicators=core_indicators,
                recommendations=core_recs,
            ))

        return details

    def _analyze_bad_speculation(self, fv: WorkloadFeatureVector) -> List[BottleneckDetail]:
        """Analyze bad speculation bottleneck."""
        details = []
        bad = fv.compute.topdown_l1.bad_speculation

        if bad < self.BAD_SPEC_THRESHOLD:
            return details

        indicators = []
        recommendations = []

        if fv.branch.branch_mpki > self.HIGH_BRANCH_MPKI:
            indicators.append(
                f"Branch MPKI = {fv.branch.branch_mpki:.1f} — high misprediction "
                f"rate wasting pipeline slots"
            )

        if (fv.branch.indirect_branch_ratio is not None
                and fv.branch.indirect_branch_ratio > self.HIGH_INDIRECT_RATIO):
            indicators.append(
                f"Indirect branch ratio = {fv.branch.indirect_branch_ratio:.0%} "
                f"— virtual dispatch / function pointer heavy code"
            )
            recommendations.append(
                "Larger BTB with better indirect branch predictor (e.g., ITTAGE)"
            )

        if fv.branch.branch_mispredict_rate > 0.05:
            indicators.append(
                f"Mispredict rate = {fv.branch.branch_mispredict_rate:.1%}"
            )
            recommendations.append(
                "Improved branch predictor (larger TAGE tables, longer history)"
            )

        if not indicators:
            indicators.append("Bad speculation elevated but root cause unclear")

        details.append(BottleneckDetail(
            category="Bad Speculation",
            score=bad,
            indicators=indicators,
            recommendations=recommendations,
        ))

        return details

    def _generate_architect_notes(
        self,
        fv: WorkloadFeatureVector,
        primary: BottleneckCategory,
        details: List[BottleneckDetail],
    ) -> List[str]:
        """Generate human-readable notes for chip architects."""
        notes = []

        notes.append(
            f"Workload '{fv.scenario_name}' ({fv.scenario_type.value}) is primarily "
            f"{primary.value.replace('_', ' ')} "
            f"(TopDown: FE={fv.compute.topdown_l1.frontend_bound:.0%}, "
            f"BE={fv.compute.topdown_l1.backend_bound:.0%}, "
            f"RET={fv.compute.topdown_l1.retiring:.0%}, "
            f"BS={fv.compute.topdown_l1.bad_speculation:.0%})"
        )

        notes.append(
            f"IPC = {fv.compute.ipc:.2f}, CPI = {fv.compute.cpi:.2f} "
            f"(dispatch width = {self.dispatch_width}, utilization = {fv.compute.ipc/self.dispatch_width:.0%})"
        )

        # Cache hierarchy summary
        notes.append(
            f"Cache MPKI waterfall: L1I={fv.cache.l1i_mpki:.1f}, "
            f"L1D={fv.cache.l1d_mpki:.1f}, L2={fv.cache.l2_mpki:.1f}, "
            f"L3={fv.cache.l3_mpki:.1f}"
        )

        # Key design takeaway
        for detail in details:
            if detail.recommendations:
                notes.append(f"[{detail.category}] {detail.recommendations[0]}")

        return notes

    def _generate_summary(self, report: BottleneckReport) -> str:
        """One-line summary for the bottleneck."""
        return (
            f"{report.primary_bottleneck.value.replace('_', ' ').title()} "
            f"({report.primary_score:.0%})"
        )
