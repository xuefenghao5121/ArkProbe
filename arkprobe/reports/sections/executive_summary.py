"""Executive summary section renderer."""
from __future__ import annotations
from typing import Dict, List
from ...model.schema import WorkloadFeatureVector
from ...analysis.bottleneck_analyzer import BottleneckReport
from ...analysis.design_space import DesignSensitivityReport


def render_executive_summary(
    feature_vectors: List[WorkloadFeatureVector],
    bottleneck_reports: Dict[str, BottleneckReport],
    sensitivity_report: DesignSensitivityReport,
) -> str:
    """Render the executive summary section."""
    # Build summary table rows
    rows = ""
    for fv in feature_vectors:
        br = bottleneck_reports.get(fv.scenario_name)
        bottleneck = br.summary if br else "N/A"

        # Get top sensitivity
        scores = sensitivity_report.per_workload.get(fv.scenario_name, [])
        top_param = max(scores, key=lambda s: s.score) if scores else None
        top_sens = f"{top_param.parameter} ({top_param.score:.0%})" if top_param else "N/A"

        rows += f"""<tr>
            <td>{fv.scenario_name}</td>
            <td>{fv.scenario_type.value}</td>
            <td>{fv.compute.ipc:.2f}</td>
            <td>{bottleneck}</td>
            <td>{top_sens}</td>
        </tr>"""

    # Top recommendations
    recs_html = ""
    for i, rec in enumerate(sensitivity_report.recommendations[:5]):
        recs_html += f"""<tr>
            <td>{i+1}</td>
            <td>{rec.parameter}</td>
            <td>{rec.priority:.3f}</td>
            <td><span class="tag tag-{rec.area_cost}">{rec.area_cost}</span></td>
            <td>{len(rec.affected_workloads)} workloads</td>
        </tr>"""

    return f"""
    <div class="section" id="executive-summary">
        <h2>1. Executive Summary</h2>
        <p>Analysis of {len(feature_vectors)} workload scenarios profiled on Kunpeng ARM platform.
        The table below summarizes each workload's primary micro-architectural bottleneck
        and the hardware design parameter it is most sensitive to.</p>

        <h3>Workload Overview</h3>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Type</th>
                    <th>IPC</th>
                    <th>Primary Bottleneck</th>
                    <th>Top Sensitivity</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>

        <h3>Top Design Priorities (Cost-Adjusted)</h3>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Parameter</th>
                    <th>Priority</th>
                    <th>Area Cost</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>{recs_html}</tbody>
        </table>
    </div>"""
