"""Design recommendations section renderer."""
from __future__ import annotations
from typing import List
from ...model.schema import WorkloadFeatureVector
from ...analysis.design_space import DesignSensitivityReport
from ..charts import ChartFactory


def render_design_recommendations(
    feature_vectors: List[WorkloadFeatureVector],
    sensitivity_report: DesignSensitivityReport,
) -> str:
    """Render the hardware design recommendations section."""
    # Sensitivity heatmap
    heatmap_html = ""
    if sensitivity_report.matrix is not None:
        heatmap_fig = ChartFactory.sensitivity_heatmap(sensitivity_report.matrix)
        heatmap_html = heatmap_fig.to_html(include_plotlyjs=False, full_html=False)

    # Recommendation bar chart
    rec_fig = ChartFactory.recommendation_bar(sensitivity_report.recommendations)
    rec_html = rec_fig.to_html(include_plotlyjs=False, full_html=False)

    # Detailed recommendations table
    detail_rows = ""
    for i, rec in enumerate(sensitivity_report.recommendations):
        detail_rows += f"""<tr>
            <td>{i+1}</td>
            <td><strong>{rec.parameter}</strong></td>
            <td>{rec.direction}</td>
            <td>{rec.priority:.3f}</td>
            <td><span class="tag tag-{rec.area_cost}">{rec.area_cost}</span></td>
            <td>{rec.justification}</td>
        </tr>"""

    return f"""
    <div class="section" id="design-recommendations">
        <h2>4. Hardware Design Recommendations</h2>

        <h3>Sensitivity Matrix</h3>
        <p>Each cell shows how sensitive a workload is to a design parameter (0 = insensitive, 1 = highly sensitive).
        Use this matrix to identify which parameters benefit the most workloads.</p>
        <div class="chart-container">{heatmap_html}</div>

        <h3>Ranked Recommendations</h3>
        <p>Parameters ranked by cost-adjusted priority: higher scores indicate better ROI
        (more workloads benefit relative to the silicon area cost).</p>
        <div class="chart-container">{rec_html}</div>

        <h3>Detailed Recommendation Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Parameter</th><th>Direction</th>
                    <th>Priority</th><th>Area Cost</th><th>Justification</th>
                </tr>
            </thead>
            <tbody>{detail_rows}</tbody>
        </table>
    </div>"""
