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

    # Thresholds for bottleneck classification (based on ARM TopDown best practices)
    # A component is considered a bottleneck if it exceeds these thresholds
    FRONTEND_THRESHOLD = 0.15  # Frontend stalls > 15% is significant
    BACKEND_THRESHOLD = 0.25   # Backend stalls > 25% is significant
    BAD_SPEC_THRESHOLD = 0.10  # Bad speculation > 10% is significant
    RETIRING_GOOD_THRESHOLD = 0.50  # Retiring > 50% means well-balanced

    # Sub-thresholds for detailed indicators
    HIGH_L3_MPKI = 5.0         # L3 misses per 1000 instructions
    HIGH_L2_MPKI = 10.0        # L2 misses per 1000 instructions
    HIGH_L1D_MPKI = 30.0       # L1D misses per 1000 instructions
    HIGH_BRANCH_MPKI = 5.0     # Branch mispredictions per 1000 instructions
    HIGH_L1I_MPKI = 3.0        # L1I misses per 1000 instructions
    HIGH_TLB_MPKI = 1.0        # TLB misses per 1000 instructions
    HIGH_BW_UTIL = 0.60        # Memory bandwidth utilization > 60% is high
    CRITICAL_BW_UTIL = 0.85    # Memory bandwidth > 85% is critical saturation
    HIGH_INDIRECT_RATIO = 0.3  # Indirect branch ratio > 30%
    LOW_IPC_THRESHOLD = 1.0    # IPC < 1.0 indicates severe stall

    # JVM bottleneck thresholds
    JVM_GC_PAUSE_RATIO_HIGH = 0.10      # GC pause > 10% of total time
    JVM_GC_PAUSE_RATIO_MEDIUM = 0.05    # GC pause > 5% is potential bottleneck
    JVM_SAFEPPOINT_RATIO_HIGH = 0.05    # Safepoint > 5% is significant
    JVM_HEAP_USAGE_HIGH = 0.85          # Heap usage > 85% is pressure
    JVM_DEOPT_RATIO_HIGH = 0.10         # Deopt ratio > 10% is abnormal

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

        # JVM-specific bottleneck analysis
        if fv.jvm is not None:
            jvm_details = self._analyze_jvm_bottlenecks(fv)
            details.extend(jvm_details)
            # Override primary if JVM bottleneck is severe
            if jvm_details:
                worst = max(jvm_details, key=lambda d: d.score)
                if worst.score > self._primary_score(td, primary):
                    # Map JVM detail category to BottleneckCategory
                    jvm_cat_map = {
                        "JVM: GC Pause": BottleneckCategory.JVM_GC_PAUSE,
                        "JVM: Safepoint": BottleneckCategory.JVM_SAFEPPOINT,
                        "JVM: JIT Deoptimization": BottleneckCategory.JVM_JIT_DEOPT,
                        "JVM: Heap Pressure": BottleneckCategory.JVM_HEAP_PRESSURE,
                    }
                    mapped = jvm_cat_map.get(worst.category, primary)
                    if mapped != primary:
                        primary = mapped

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
        """Determine the primary bottleneck category using TopDown hierarchy.

        Priority order: Backend > Frontend > Bad Speculation > Well Balanced
        Backend is further split into Memory-Bound vs Core-Bound.
        """
        # If retiring is high, workload is well-balanced (no significant bottleneck)
        if ret >= self.RETIRING_GOOD_THRESHOLD:
            return BottleneckCategory.WELL_BALANCED

        # Backend is the most common bottleneck - check first
        if be >= self.BACKEND_THRESHOLD:
            # Distinguish Memory-Bound vs Core-Bound
            # Memory-bound: high L3 MPKI, high bandwidth utilization, or high L2 MPKI
            # Core-bound: high IPC utilization but still backend-bound (execution unit saturation)
            # This distinction requires feature vector context, so we default to MEMORY
            # and refine in _analyze_backend()
            return BottleneckCategory.BACKEND_MEMORY_BOUND

        # Frontend bottleneck: instruction fetch/decode stalls
        if fe >= self.FRONTEND_THRESHOLD:
            return BottleneckCategory.FRONTEND_BOUND

        # Bad speculation: branch mispredictions wasting pipeline slots
        if bad >= self.BAD_SPEC_THRESHOLD:
            return BottleneckCategory.BAD_SPECULATION

        # No dominant bottleneck, but retiring is low - classify as backend memory
        # (most common case for memory-intensive workloads)
        if be > fe and be > bad:
            return BottleneckCategory.BACKEND_MEMORY_BOUND

        return BottleneckCategory.WELL_BALANCED

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
        """Analyze backend bottleneck: split into memory-bound and core-bound.

        Memory-Bound indicators:
        - High L3 MPKI (working set exceeds LLC)
        - High L2 MPKI (working set exceeds L1D)
        - High memory bandwidth utilization
        - Low NUMA local ratio

        Core-Bound indicators:
        - High IPC utilization but still backend-bound
        - High FP/SIMD ratio (execution unit saturation)
        - No significant memory pressure
        """
        details = []
        be = fv.compute.topdown_l1.backend_bound

        if be < self.BACKEND_THRESHOLD * 0.5:
            return details

        # Calculate memory pressure score (0-1)
        memory_pressure_score = 0.0
        memory_indicators = []
        memory_recs = []

        # L3 cache pressure (most important for memory-bound classification)
        if fv.cache.l3_mpki > self.HIGH_L3_MPKI:
            l3_severity = min(1.0, fv.cache.l3_mpki / 20.0)  # 20 MPKI = max severity
            memory_pressure_score = max(memory_pressure_score, l3_severity)
            memory_indicators.append(
                f"L3 MPKI = {fv.cache.l3_mpki:.1f} (threshold: {self.HIGH_L3_MPKI}) — "
                f"working set exceeds LLC capacity, causing {fv.cache.l3_miss_rate:.1%} miss rate"
            )
            # Estimate potential L3 size increase benefit
            if fv.cache.l3_mpki > 15:
                memory_recs.append(
                    f"CRITICAL: L3 cache size increase highly recommended. "
                    f"Current L3 MPKI ({fv.cache.l3_mpki:.1f}) indicates severe cache pressure."
                )
            else:
                memory_recs.append(
                    f"Increasing L3 cache size or associativity could reduce backend stalls. "
                    f"Estimated benefit: medium-high"
                )

        # L2 cache pressure
        if fv.cache.l2_mpki > self.HIGH_L2_MPKI:
            memory_pressure_score = max(memory_pressure_score, fv.cache.l2_mpki / 30.0)
            memory_indicators.append(
                f"L2 MPKI = {fv.cache.l2_mpki:.1f} — significant L2 misses, "
                f"working set exceeds L1D capacity"
            )
            memory_recs.append("Consider larger L2 cache or improved L2 prefetcher")

        # L1D cache pressure
        if fv.cache.l1d_mpki > self.HIGH_L1D_MPKI:
            memory_indicators.append(
                f"L1D MPKI = {fv.cache.l1d_mpki:.1f} — high L1D miss rate"
            )
            memory_recs.append("L1D size increase or better prefetching may help")

        # Memory bandwidth saturation
        if fv.memory.bandwidth_utilization > self.HIGH_BW_UTIL:
            bw_severity = (fv.memory.bandwidth_utilization - self.HIGH_BW_UTIL) / (1.0 - self.HIGH_BW_UTIL)
            memory_pressure_score = max(memory_pressure_score, bw_severity)

            if fv.memory.bandwidth_utilization > self.CRITICAL_BW_UTIL:
                memory_indicators.append(
                    f"CRITICAL: Memory bandwidth utilization = {fv.memory.bandwidth_utilization:.0%} "
                    f"— approaching saturation, memory controller is bottleneck"
                )
                memory_recs.append(
                    "URGENT: More memory channels, higher-frequency DIMMs, or HBM required"
                )
            else:
                memory_indicators.append(
                    f"Memory bandwidth utilization = {fv.memory.bandwidth_utilization:.0%} "
                    f"— significant pressure on memory subsystem"
                )
                memory_recs.append(
                    "Additional memory channels or faster DIMMs recommended"
                )

        # NUMA remote access penalty
        if fv.memory.numa_local_ratio is not None and fv.memory.numa_local_ratio < 0.8:
            numa_penalty = (1.0 - fv.memory.numa_local_ratio) * 2  # Scale to 0-1
            memory_pressure_score = max(memory_pressure_score, numa_penalty)
            memory_indicators.append(
                f"NUMA local access ratio = {fv.memory.numa_local_ratio:.0%} "
                f"— {100 - fv.memory.numa_local_ratio:.0f}% remote memory traffic adding latency"
            )
            memory_recs.append(
                "NUMA-aware memory allocation or interconnect bandwidth improvements"
            )

        # Add Memory-Bound detail if indicators exist
        if memory_indicators:
            details.append(BottleneckDetail(
                category="Backend: Memory Bound",
                score=be * max(0.5, memory_pressure_score),  # Weight by memory pressure
                indicators=memory_indicators,
                recommendations=memory_recs,
            ))

        # Core-Bound analysis (execution unit saturation)
        core_indicators = []
        core_recs = []

        ipc_ratio = fv.compute.ipc / max(1, self.dispatch_width)

        # Core-bound: high IPC utilization but still backend-bound with low memory pressure
        if ipc_ratio > 0.6 and memory_pressure_score < 0.3:
            core_indicators.append(
                f"IPC = {fv.compute.ipc:.2f} ({ipc_ratio:.0%} of dispatch width) — "
                f"execution units are saturated"
            )
            core_recs.append("Wider dispatch width or more execution units")

        # FP/SIMD saturation
        if fv.compute.instruction_mix.fp_ratio > 0.2:
            core_indicators.append(
                f"FP instruction ratio = {fv.compute.instruction_mix.fp_ratio:.0%} "
                f"— FP/SIMD execution units may be bottleneck"
            )
            if fv.compute.instruction_mix.vector_ratio > 0.1:
                core_recs.append("Additional SIMD/NEON execution units or wider vector width")
            else:
                core_recs.append("Additional FP execution units")

        # High load/store ratio (address generation bottleneck)
        if fv.compute.instruction_mix.load_ratio + fv.compute.instruction_mix.store_ratio > 0.4:
            core_indicators.append(
                f"Load/Store ratio = {fv.compute.instruction_mix.load_ratio + fv.compute.instruction_mix.store_ratio:.0%} "
                f"— address generation units may be bottleneck"
            )
            core_recs.append("Additional load/store units or AGUs")

        if core_indicators:
            details.append(BottleneckDetail(
                category="Backend: Core Bound",
                score=be * (1 - memory_pressure_score) * ipc_ratio,
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
        """Generate human-readable notes for chip architects.

        Output format:
        1. Primary bottleneck summary with TopDown breakdown
        2. IPC/CPI analysis with dispatch utilization
        3. Cache hierarchy summary
        4. Key design recommendations
        """
        notes = []

        # 1. Primary bottleneck summary
        td = fv.compute.topdown_l1
        bottleneck_name = primary.value.replace('_', ' ').title()

        # Add severity indicator
        if primary == BottleneckCategory.WELL_BALANCED:
            severity = "✓ Good"
        elif td.backend_bound > 0.5 or td.frontend_bound > 0.4:
            severity = "⚠ CRITICAL"
        elif td.backend_bound > 0.3 or td.frontend_bound > 0.25:
            severity = "⚡ Significant"
        else:
            severity = "→ Moderate"

        notes.append(
            f"{severity}: '{fv.scenario_name}' ({fv.scenario_type.value}) is "
            f"{bottleneck_name}"
        )

        # TopDown breakdown with visual bar
        fe_bar = "█" * int(td.frontend_bound * 10)
        be_bar = "█" * int(td.backend_bound * 10)
        ret_bar = "█" * int(td.retiring * 10)
        bad_bar = "█" * int(td.bad_speculation * 10)
        notes.append(
            f"TopDown L1: FE={td.frontend_bound:.0%}{fe_bar} | "
            f"BE={td.backend_bound:.0%}{be_bar} | "
            f"RET={td.retiring:.0%}{ret_bar} | "
            f"BAD={td.bad_speculation:.0%}{bad_bar}"
        )

        # 2. IPC/CPI analysis
        ipc_util = fv.compute.ipc / self.dispatch_width
        if ipc_util > 0.7:
            ipc_status = "High utilization"
        elif ipc_util > 0.4:
            ipc_status = "Moderate utilization"
        else:
            ipc_status = "Low utilization (stalled)"

        notes.append(
            f"IPC = {fv.compute.ipc:.2f}, CPI = {fv.compute.cpi:.2f} "
            f"({ipc_status}: {ipc_util:.0%} of dispatch width {self.dispatch_width})"
        )

        # 3. Cache hierarchy summary with severity indicators
        cache_summary = []
        if fv.cache.l3_mpki > self.HIGH_L3_MPKI:
            cache_summary.append(f"L3={fv.cache.l3_mpki:.1f}⚠")
        else:
            cache_summary.append(f"L3={fv.cache.l3_mpki:.1f}")

        if fv.cache.l2_mpki > self.HIGH_L2_MPKI:
            cache_summary.append(f"L2={fv.cache.l2_mpki:.1f}⚠")
        else:
            cache_summary.append(f"L2={fv.cache.l2_mpki:.1f}")

        cache_summary.append(f"L1D={fv.cache.l1d_mpki:.1f}")
        cache_summary.append(f"L1I={fv.cache.l1i_mpki:.1f}")

        notes.append(f"Cache MPKI waterfall: {' → '.join(cache_summary)}")

        # 4. Top recommendation from each detail
        for detail in details:
            if detail.recommendations:
                # Take first recommendation, truncate if too long
                rec = detail.recommendations[0]
                if len(rec) > 100:
                    rec = rec[:97] + "..."
                notes.append(f"[{detail.category}] {rec}")

        return notes

    def _generate_summary(self, report: BottleneckReport) -> str:
        """One-line summary for the bottleneck."""
        return (
            f"{report.primary_bottleneck.value.replace('_', ' ').title()} "
            f"({report.primary_score:.0%})"
        )

    # ------------------------------------------------------------------
    # JVM bottleneck analysis
    # ------------------------------------------------------------------

    def _analyze_jvm_bottlenecks(self, fv: WorkloadFeatureVector) -> List[BottleneckDetail]:
        """Detect JVM-layer bottlenecks from JFR/jstat data."""
        if fv.jvm is None:
            return []

        details: List[BottleneckDetail] = []
        gc = fv.jvm.gc
        threads = fv.jvm.threads
        jit = fv.jvm.jit

        # GC pause bottleneck
        if gc.gc_pause_ratio >= self.JVM_GC_PAUSE_RATIO_HIGH:
            indicators = [
                f"GC pause ratio = {gc.gc_pause_ratio:.1%} "
                f"(young={gc.young_gc_count}, full={gc.full_gc_count})",
                f"GC algorithm: {gc.gc_algorithm}",
                f"Average GC pause: {gc.avg_gc_pause_ms:.1f}ms",
            ]
            recs = [
                "Consider GC algorithm change (G1→ZGC for low-latency, Parallel for throughput)",
                "Tune heap sizing: -Xms/-Xmx, -XX:NewRatio",
                "Reduce object allocation rate to lower GC pressure",
            ]
            if gc.full_gc_count > 0:
                indicators.append(f"Full GC count = {gc.full_gc_count} — potential OOM risk")
                recs.append("Investigate Full GC triggers: heap too small or metaspace exhaustion")
            details.append(BottleneckDetail(
                category="JVM: GC Pause",
                score=gc.gc_pause_ratio,
                indicators=indicators,
                recommendations=recs,
            ))
        elif gc.gc_pause_ratio >= self.JVM_GC_PAUSE_RATIO_MEDIUM:
            details.append(BottleneckDetail(
                category="JVM: GC Pause",
                score=gc.gc_pause_ratio * 0.5,
                indicators=[f"GC pause ratio = {gc.gc_pause_ratio:.1%} (moderate)"],
                recommendations=["Monitor GC trends; consider heap tuning if ratio grows"],
            ))

        # Heap pressure
        if gc.heap_usage_ratio >= self.JVM_HEAP_USAGE_HIGH:
            details.append(BottleneckDetail(
                category="JVM: Heap Pressure",
                score=gc.heap_usage_ratio,
                indicators=[
                    f"Heap usage = {gc.heap_usage_ratio:.0%} "
                    f"({gc.heap_used_mb:.0f}/{gc.heap_max_mb:.0f} MB)",
                    f"Metaspace used = {gc.metaspace_used_mb:.1f} MB",
                ],
                recommendations=[
                    "Increase max heap (-Xmx) or reduce object retention",
                    "Profile for memory leaks if usage grows over time",
                ],
            ))

        # Safepoint bottleneck
        if threads.safepoint_ratio >= self.JVM_SAFEPPOINT_RATIO_HIGH:
            details.append(BottleneckDetail(
                category="JVM: Safepoint",
                score=threads.safepoint_ratio,
                indicators=[
                    f"Safepoint ratio = {threads.safepoint_ratio:.1%} "
                    f"(count={threads.safepoint_count}, total={threads.safepoint_total_ms:.1f}ms)",
                ],
                recommendations=[
                    "Reduce safepoint triggers: disable biased locking (-XX:-UseBiasedLocking)",
                    "Use -XX:+UseCountedLoopSafepoints for long-running loops",
                ],
            ))

        # JIT deoptimization
        if jit.deopt_ratio >= self.JVM_DEOPT_RATIO_HIGH and jit.total_compilations > 0:
            details.append(BottleneckDetail(
                category="JVM: JIT Deoptimization",
                score=jit.deopt_ratio,
                indicators=[
                    f"Deoptimization ratio = {jit.deopt_ratio:.1%} "
                    f"({jit.deoptimization_count}/{jit.total_compilations})",
                ],
                recommendations=[
                    "Review hot method profiles for unstable type assumptions",
                    "Consider -XX:CompileThreshold increase to delay compilation",
                ],
            ))

        return details
