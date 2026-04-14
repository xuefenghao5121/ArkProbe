"""Comparator for tuning experiment results.

Compares performance impact across different hardware configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..model.feature_vector import WorkloadFeatureVector

logger = logging.getLogger(__name__)


@dataclass
class MetricChange:
    """Change in a single metric between configurations.

    Attributes:
        name: Metric name
        baseline_value: Value in baseline configuration
        tuned_value: Value in tuned configuration
        absolute_change: Absolute difference (tuned - baseline)
        percent_change: Percentage change
    """
    name: str
    baseline_value: float
    tuned_value: float
    absolute_change: float
    percent_change: float

    @property
    def improved(self) -> bool:
        """Whether the change represents an improvement.

        For IPC, higher is better.
        For MPKI, lower is better.
        """
        if self.name in ("ipc", "retiring", "simd_utilization"):
            return self.percent_change > 0
        elif self.name in ("l1i_mpki", "l1d_mpki", "l2_mpki", "l3_mpki",
                           "branch_mpki", "tlb_mpki"):
            return self.percent_change < 0
        else:
            return self.percent_change > 0


@dataclass
class BottleneckShift:
    """Shift in bottleneck classification.

    Attributes:
        baseline_bottleneck: Primary bottleneck in baseline
        tuned_bottleneck: Primary bottleneck after tuning
        severity_change: Change in bottleneck severity
    """
    baseline_bottleneck: str
    tuned_bottleneck: str
    severity_change: float


@dataclass
class ImpactReport:
    """Report comparing performance between two configurations.

    Attributes:
        config_name: Name of the tuned configuration
        baseline_name: Name of the baseline configuration
        metric_changes: Changes in key metrics
        bottleneck_shift: Shift in primary bottleneck
        overall_improvement: Overall improvement score (-1 to 1)
        key_findings: List of key findings
        recommendations: Recommendations based on results
    """
    config_name: str
    baseline_name: str
    metric_changes: list[MetricChange] = field(default_factory=list)
    bottleneck_shift: Optional[BottleneckShift] = None
    overall_improvement: float = 0.0
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config_name,
            "baseline_name": self.baseline_name,
            "metric_changes": [
                {
                    "name": c.name,
                    "baseline": c.baseline_value,
                    "tuned": c.tuned_value,
                    "change_pct": c.percent_change,
                    "improved": c.improved,
                }
                for c in self.metric_changes
            ],
            "bottleneck_shift": {
                "baseline": self.bottleneck_shift.baseline_bottleneck,
                "tuned": self.bottleneck_shift.tuned_bottleneck,
                "severity_change": self.bottleneck_shift.severity_change,
            } if self.bottleneck_shift else None,
            "overall_improvement": self.overall_improvement,
            "key_findings": self.key_findings,
            "recommendations": self.recommendations,
        }


class TuningComparator:
    """Compare performance across different tuning configurations.

    Usage:
        comparator = TuningComparator()

        # Compare two feature vectors
        report = comparator.compare(baseline_fv, tuned_fv)

        # Compare multiple configurations
        reports = comparator.compare_all(baseline_fv, [tuned1_fv, tuned2_fv])
    """

    # Metrics to compare (name, attribute path, higher_is_better)
    METRICS_TO_COMPARE = [
        ("ipc", "compute.ipc", True),
        ("retiring", "compute.topdown_l1.retiring", True),
        ("frontend_bound", "compute.topdown_l1.frontend_bound", False),
        ("backend_bound", "compute.topdown_l1.backend_bound", False),
        ("bad_speculation", "compute.topdown_l1.bad_speculation", False),
        ("l1i_mpki", "cache.l1i_mpki", False),
        ("l1d_mpki", "cache.l1d_mpki", False),
        ("l2_mpki", "cache.l2_mpki", False),
        ("l3_mpki", "cache.l3_mpki", False),
        ("branch_mpki", "branch.branch_mpki", False),
        ("tlb_mpki", "memory.tlb_mpki", False),
    ]

    def compare(
        self,
        baseline: WorkloadFeatureVector,
        tuned: WorkloadFeatureVector,
        config_name: Optional[str] = None,
    ) -> ImpactReport:
        """Compare tuned configuration against baseline.

        Args:
            baseline: Feature vector from baseline configuration
            tuned: Feature vector from tuned configuration
            config_name: Name of the tuned configuration

        Returns:
            ImpactReport with detailed comparison
        """
        config_name = config_name or tuned.scenario_name
        metric_changes = []

        for name, path, _ in self.METRICS_TO_COMPARE:
            baseline_val = self._get_nested_attr(baseline, path)
            tuned_val = self._get_nested_attr(tuned, path)

            if baseline_val is None or tuned_val is None:
                continue

            if baseline_val == 0:
                pct_change = 0.0 if tuned_val == 0 else float('inf')
            else:
                pct_change = ((tuned_val - baseline_val) / baseline_val) * 100

            metric_changes.append(MetricChange(
                name=name,
                baseline_value=baseline_val,
                tuned_value=tuned_val,
                absolute_change=tuned_val - baseline_val,
                percent_change=pct_change,
            ))

        # Analyze bottleneck shift
        bottleneck_shift = self._analyze_bottleneck_shift(baseline, tuned)

        # Calculate overall improvement score
        overall = self._calculate_overall_improvement(metric_changes)

        # Generate findings and recommendations
        findings = self._generate_findings(metric_changes, bottleneck_shift)
        recommendations = self._generate_recommendations(metric_changes, bottleneck_shift)

        return ImpactReport(
            config_name=config_name,
            baseline_name=baseline.scenario_name,
            metric_changes=metric_changes,
            bottleneck_shift=bottleneck_shift,
            overall_improvement=overall,
            key_findings=findings,
            recommendations=recommendations,
        )

    def compare_all(
        self,
        baseline: WorkloadFeatureVector,
        tuned_configs: list[tuple[str, WorkloadFeatureVector]],
    ) -> list[ImpactReport]:
        """Compare multiple tuned configurations against baseline.

        Args:
            baseline: Feature vector from baseline configuration
            tuned_configs: List of (config_name, feature_vector) tuples

        Returns:
            List of ImpactReports for each configuration
        """
        return [
            self.compare(baseline, fv, name)
            for name, fv in tuned_configs
        ]

    def _get_nested_attr(self, obj, path: str) -> Optional[float]:
        """Get nested attribute by dot-separated path."""
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return float(obj) if obj is not None else None
        except (AttributeError, TypeError, ValueError):
            return None

    def _analyze_bottleneck_shift(
        self,
        baseline: WorkloadFeatureVector,
        tuned: WorkloadFeatureVector,
    ) -> Optional[BottleneckShift]:
        """Analyze shift in primary bottleneck."""
        # Get primary bottleneck from TopDown
        baseline_td = baseline.compute.topdown_l1
        tuned_td = tuned.compute.topdown_l1

        # Find primary bottleneck (highest bound)
        baseline_bounds = {
            "Frontend": baseline_td.frontend_bound,
            "Backend": baseline_td.backend_bound,
            "BadSpec": baseline_td.bad_speculation,
        }
        tuned_bounds = {
            "Frontend": tuned_td.frontend_bound,
            "Backend": tuned_td.backend_bound,
            "BadSpec": tuned_td.bad_speculation,
        }

        baseline_bottleneck = max(baseline_bounds, key=baseline_bounds.get)
        tuned_bottleneck = max(tuned_bounds, key=tuned_bounds.get)

        severity_change = (
            tuned_bounds[tuned_bottleneck] - baseline_bounds[baseline_bottleneck]
        )

        return BottleneckShift(
            baseline_bottleneck=baseline_bottleneck,
            tuned_bottleneck=tuned_bottleneck,
            severity_change=severity_change,
        )

    def _calculate_overall_improvement(self, changes: list[MetricChange]) -> float:
        """Calculate overall improvement score.

        Returns:
            Score from -1 (much worse) to 1 (much better)
        """
        if not changes:
            return 0.0

        # Weight key metrics
        weights = {
            "ipc": 3.0,
            "l3_mpki": 2.0,
            "branch_mpki": 1.5,
            "retiring": 2.0,
            "backend_bound": 1.5,
        }

        weighted_sum = 0.0
        weight_total = 0.0

        for change in changes:
            weight = weights.get(change.name, 1.0)
            # Normalize percent change to [-1, 1]
            normalized = max(-1, min(1, change.percent_change / 50))

            # Invert for metrics where lower is better
            if change.name in ("l1i_mpki", "l1d_mpki", "l2_mpki", "l3_mpki",
                               "branch_mpki", "tlb_mpki", "frontend_bound",
                               "backend_bound", "bad_speculation"):
                normalized = -normalized

            weighted_sum += weight * normalized
            weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _generate_findings(
        self,
        changes: list[MetricChange],
        bottleneck_shift: Optional[BottleneckShift],
    ) -> list[str]:
        """Generate key findings from the comparison."""
        findings = []

        # IPC change
        ipc_change = next((c for c in changes if c.name == "ipc"), None)
        if ipc_change:
            if ipc_change.percent_change > 5:
                findings.append(
                    f"IPC improved by {ipc_change.percent_change:.1f}% "
                    f"({ipc_change.baseline_value:.2f} → {ipc_change.tuned_value:.2f})"
                )
            elif ipc_change.percent_change < -5:
                findings.append(
                    f"IPC degraded by {abs(ipc_change.percent_change):.1f}% "
                    f"({ipc_change.baseline_value:.2f} → {ipc_change.tuned_value:.2f})"
                )

        # Cache MPKI changes
        for name in ("l3_mpki", "l2_mpki", "l1d_mpki"):
            change = next((c for c in changes if c.name == name), None)
            if change and abs(change.percent_change) > 10:
                direction = "reduced" if change.percent_change < 0 else "increased"
                findings.append(
                    f"{name.upper()} {direction} by {abs(change.percent_change):.1f}%"
                )

        # Branch MPKI
        branch_change = next((c for c in changes if c.name == "branch_mpki"), None)
        if branch_change and abs(branch_change.percent_change) > 10:
            direction = "reduced" if branch_change.percent_change < 0 else "increased"
            findings.append(
                f"Branch mispredictions {direction} by {abs(branch_change.percent_change):.1f}%"
            )

        # Bottleneck shift
        if bottleneck_shift:
            if bottleneck_shift.baseline_bottleneck != bottleneck_shift.tuned_bottleneck:
                findings.append(
                    f"Bottleneck shifted from {bottleneck_shift.baseline_bottleneck} "
                    f"to {bottleneck_shift.tuned_bottleneck}"
                )
            elif bottleneck_shift.severity_change < -0.1:
                findings.append(
                    f"{bottleneck_shift.tuned_bottleneck} bottleneck reduced "
                    f"by {abs(bottleneck_shift.severity_change):.0%}"
                )

        return findings

    def _generate_recommendations(
        self,
        changes: list[MetricChange],
        bottleneck_shift: Optional[BottleneckShift],
    ) -> list[str]:
        """Generate recommendations based on results."""
        recommendations = []

        # Check if tuning was effective
        ipc_change = next((c for c in changes if c.name == "ipc"), None)
        l3_change = next((c for c in changes if c.name == "l3_mpki"), None)

        if ipc_change and ipc_change.percent_change > 10:
            recommendations.append(
                "Configuration shows significant IPC improvement - "
                "consider using for similar workloads"
            )

        if l3_change and l3_change.percent_change < -15:
            recommendations.append(
                "L3 cache behavior improved - configuration benefits memory-bound workloads"
            )

        # Check for regressions
        regressions = [c for c in changes if not c.improved and abs(c.percent_change) > 5]
        if regressions:
            recommendations.append(
                f"Warning: {len(regressions)} metrics regressed - "
                "evaluate trade-offs for your workload"
            )

        # Bottleneck-specific recommendations
        if bottleneck_shift:
            if bottleneck_shift.tuned_bottleneck == "Backend":
                recommendations.append(
                    "Workload is Backend-bound - consider memory/execution unit tuning"
                )
            elif bottleneck_shift.tuned_bottleneck == "Frontend":
                recommendations.append(
                    "Workload is Frontend-bound - consider instruction cache/branch tuning"
                )
            elif bottleneck_shift.tuned_bottleneck == "BadSpec":
                recommendations.append(
                    "Workload has high speculation overhead - consider branch prediction tuning"
                )

        return recommendations
