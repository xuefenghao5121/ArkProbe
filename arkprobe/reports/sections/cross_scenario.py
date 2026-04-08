"""Cross-scenario comparison section renderer."""
from __future__ import annotations
from typing import Dict, List
from ...model.schema import WorkloadFeatureVector
from ...analysis.comparator import ComparisonReport
from ..charts import ChartFactory


def render_cross_scenario(
    feature_vectors: List[WorkloadFeatureVector],
    comparison: ComparisonReport,
) -> str:
    """Render the cross-scenario comparison section."""
    # TopDown comparison
    topdown_fig = ChartFactory.topdown_comparison(feature_vectors)
    topdown_html = topdown_fig.to_html(include_plotlyjs=False, full_html=False)

    # Radar chart
    radar_fig = ChartFactory.radar_overlay(feature_vectors, comparison.radar_data)
    radar_html = radar_fig.to_html(include_plotlyjs=False, full_html=False)

    # PCA scatter
    pca_html = ""
    if comparison.pca_data and comparison.pca_data.get("points"):
        pca_fig = ChartFactory.pca_scatter(comparison.pca_data)
        pca_html = f'<div class="chart-container">{pca_fig.to_html(include_plotlyjs=False, full_html=False)}</div>'

    # BW-Latency scatter
    bwlat_fig = ChartFactory.bandwidth_latency_scatter(feature_vectors)
    bwlat_html = bwlat_fig.to_html(include_plotlyjs=False, full_html=False)

    # Scalability comparison
    scale_fig = ChartFactory.scalability_lines(feature_vectors)
    scale_html = scale_fig.to_html(include_plotlyjs=False, full_html=False)

    # Cluster summary
    cluster_html = ""
    if comparison.clusters:
        cluster_html = "<h3>Workload Clusters</h3><p>Workloads grouped by micro-architectural similarity:</p>"
        for cluster_id, members in comparison.clusters.items():
            cluster_html += f'<p><strong>Cluster {cluster_id}:</strong> {", ".join(members)}</p>'

    return f"""
    <div class="section" id="cross-scenario">
        <h2>3. Cross-Scenario Comparison</h2>
        <div class="chart-container">{topdown_html}</div>
        <div class="grid-2">
            <div class="chart-container">{radar_html}</div>
            <div class="chart-container">{bwlat_html}</div>
        </div>
        {pca_html}
        <div class="chart-container">{scale_html}</div>
        {cluster_html}
    </div>"""
