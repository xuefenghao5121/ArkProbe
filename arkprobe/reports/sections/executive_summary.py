"""Executive summary section renderer."""
from __future__ import annotations
from typing import Dict, List
from ...model.schema import WorkloadFeatureVector
from ...analysis.bottleneck_analyzer import BottleneckReport
from ...analysis.design_space import DesignSensitivityReport


def _get_bottleneck_severity(br: BottleneckReport) -> str:
    """Get severity class for bottleneck styling."""
    score = br.primary_score
    if score >= 0.5:
        return "critical"
    elif score >= 0.3:
        return "significant"
    elif score >= 0.15:
        return "moderate"
    else:
        return "good"


def _get_ipc_status(ipc: float, dispatch_width: int = 4) -> tuple:
    """Get IPC status and color."""
    util = ipc / dispatch_width
    if util >= 0.7:
        return "High", "#2ca02c"
    elif util >= 0.4:
        return "Medium", "#ff7f0e"
    else:
        return "Low", "#d62728"


def _get_l3_status(l3_mpki: float) -> tuple:
    """Get L3 MPKI status and color."""
    if l3_mpki <= 2:
        return "Good", "#2ca02c"
    elif l3_mpki <= 5:
        return "OK", "#ffbb78"
    elif l3_mpki <= 15:
        return "High", "#ff7f0e"
    else:
        return "Critical", "#d62728"


def render_executive_summary(
    feature_vectors: List[WorkloadFeatureVector],
    bottleneck_reports: Dict[str, BottleneckReport],
    sensitivity_report: DesignSensitivityReport,
) -> str:
    """Render the executive summary section.

    Shows:
    1. Workload overview table with key metrics
    2. Top design priorities
    3. Quick insights summary
    """
    # Build summary table rows with enhanced metrics
    rows = ""
    for fv in feature_vectors:
        br = bottleneck_reports.get(fv.scenario_name)

        # Bottleneck with severity styling
        if br:
            severity = _get_bottleneck_severity(br)
            bottleneck = f'<span class="severity-{severity}">{br.summary}</span>'
        else:
            bottleneck = "N/A"

        # IPC with status
        ipc_status, ipc_color = _get_ipc_status(fv.compute.ipc)
        ipc_display = f'<span style="color:{ipc_color}">{fv.compute.ipc:.2f}</span>'

        # L3 MPKI with status
        l3_status, l3_color = _get_l3_status(fv.cache.l3_mpki)
        l3_display = f'<span style="color:{l3_color}">{fv.cache.l3_mpki:.1f}</span>'

        # Memory bandwidth utilization
        bw_util = fv.memory.bandwidth_utilization
        if bw_util > 0.8:
            bw_display = f'<span style="color:#d62728">{bw_util:.0%}</span>'
        elif bw_util > 0.5:
            bw_display = f'<span style="color:#ff7f0e">{bw_util:.0%}</span>'
        else:
            bw_display = f'{bw_util:.0%}'

        # Top sensitivity
        scores = sensitivity_report.per_workload.get(fv.scenario_name, [])
        top_param = max(scores, key=lambda s: s.score) if scores else None
        top_sens = f"{top_param.parameter} ({top_param.score:.0%})" if top_param else "N/A"

        rows += f"""<tr>
            <td><strong>{fv.scenario_name}</strong></td>
            <td>{fv.scenario_type.value}</td>
            <td>{ipc_display}</td>
            <td>{l3_display}</td>
            <td>{bw_display}</td>
            <td>{bottleneck}</td>
            <td>{top_sens}</td>
        </tr>"""

    # Top recommendations with impact visualization
    recs_html = ""
    for i, rec in enumerate(sensitivity_report.recommendations[:5]):
        # Priority bar visualization
        priority_bar = "█" * int(rec.priority * 10)
        cost_colors = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}
        cost_color = cost_colors.get(rec.area_cost, "#7f7f7f")

        recs_html += f"""<tr>
            <td>{i+1}</td>
            <td><strong>{rec.parameter}</strong></td>
            <td>
                <span style="color:{cost_color}">{rec.priority:.3f}</span>
                <span style="opacity:0.5">{priority_bar}</span>
            </td>
            <td><span class="tag tag-{rec.area_cost}">{rec.area_cost}</span></td>
            <td>{len(rec.affected_workloads)} workloads</td>
        </tr>"""

    # Quick insights
    insights = _generate_quick_insights(feature_vectors, bottleneck_reports)

    return f"""
    <div class="section" id="executive-summary">
        <h2>1. Executive Summary</h2>
        <p>Analysis of <strong>{len(feature_vectors)}</strong> workload scenarios profiled on Kunpeng ARM platform.
        Key metrics below show each workload's performance characteristics and primary bottlenecks.</p>

        {insights}

        <h3>Workload Overview</h3>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Type</th>
                    <th>IPC</th>
                    <th>L3 MPKI</th>
                    <th>Mem BW</th>
                    <th>Primary Bottleneck</th>
                    <th>Top Sensitivity</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>

        <h3>Top Design Priorities (Cost-Adjusted)</h3>
        <p>Hardware parameters ranked by cross-workload impact, adjusted for implementation cost.</p>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Parameter</th>
                    <th>Priority</th>
                    <th>Cost</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>{recs_html}</tbody>
        </table>
    </div>"""


def _generate_quick_insights(
    feature_vectors: List[WorkloadFeatureVector],
    bottleneck_reports: Dict[str, BottleneckReport],
) -> str:
    """Generate quick insights summary."""
    if not feature_vectors:
        return ""

    insights = []

    # Count bottleneck types
    bottleneck_counts = {}
    for br in bottleneck_reports.values():
        cat = br.primary_bottleneck.value
        bottleneck_counts[cat] = bottleneck_counts.get(cat, 0) + 1

    # Most common bottleneck
    if bottleneck_counts:
        top_bottleneck = max(bottleneck_counts, key=bottleneck_counts.get)
        count = bottleneck_counts[top_bottleneck]
        insights.append(
            f"📊 <strong>Most common bottleneck:</strong> {top_bottleneck.replace('_', ' ')} "
            f"({count}/{len(feature_vectors)} workloads)"
        )

    # Average IPC
    avg_ipc = sum(fv.compute.ipc for fv in feature_vectors) / len(feature_vectors)
    insights.append(f"⚡ <strong>Average IPC:</strong> {avg_ipc:.2f}")

    # Memory-bound workloads
    memory_bound = sum(
        1 for br in bottleneck_reports.values()
        if "memory" in br.primary_bottleneck.value.lower()
    )
    if memory_bound > 0:
        insights.append(
            f"💾 <strong>Memory-bound workloads:</strong> {memory_bound} — "
            f"consider cache/memory optimizations"
        )

    # High L3 MPKI workloads
    high_l3 = sum(1 for fv in feature_vectors if fv.cache.l3_mpki > 10)
    if high_l3 > 0:
        insights.append(
            f"🔥 <strong>High L3 pressure:</strong> {high_l3} workload(s) with L3 MPKI > 10"
        )

    if not insights:
        return ""

    insights_html = "<ul>" + "".join(f"<li>{i}</li>" for i in insights) + "</ul>"
    return f'<div class="insights-box"><h4>Quick Insights</h4>{insights_html}</div>'
