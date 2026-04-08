"""Per-scenario deep dive section renderer."""
from __future__ import annotations
from typing import Optional
from ...model.schema import WorkloadFeatureVector
from ...analysis.bottleneck_analyzer import BottleneckReport
from ..charts import ChartFactory


def render_scenario_section(
    fv: WorkloadFeatureVector,
    bottleneck: Optional[BottleneckReport] = None,
) -> str:
    """Render a single scenario deep-dive section."""
    # Generate charts
    topdown_fig = ChartFactory.topdown_stacked_bar(fv)
    cache_fig = ChartFactory.cache_mpki_waterfall(fv)
    instmix_fig = ChartFactory.instruction_mix_pie(fv)

    topdown_html = topdown_fig.to_html(include_plotlyjs=False, full_html=False)
    cache_html = cache_fig.to_html(include_plotlyjs=False, full_html=False)
    instmix_html = instmix_fig.to_html(include_plotlyjs=False, full_html=False)

    # Bottleneck analysis narrative
    notes_html = ""
    if bottleneck:
        for note in bottleneck.architect_notes:
            notes_html += f'<div class="note"><p>{note}</p></div>'

        for detail in bottleneck.details:
            notes_html += f'<h4>{detail.category} ({detail.score:.0%})</h4>'
            for ind in detail.indicators:
                notes_html += f'<p>- {ind}</p>'
            for rec in detail.recommendations:
                notes_html += f'<p><strong>Recommendation:</strong> {rec}</p>'

    # Key metrics cards
    metrics = f"""
    <div class="grid-3">
        <div class="metric-card">
            <div class="value">{fv.compute.ipc:.2f}</div>
            <div class="label">IPC</div>
        </div>
        <div class="metric-card">
            <div class="value">{fv.cache.l3_mpki:.1f}</div>
            <div class="label">L3 MPKI</div>
        </div>
        <div class="metric-card">
            <div class="value">{fv.branch.branch_mpki:.1f}</div>
            <div class="label">Branch MPKI</div>
        </div>
    </div>"""

    return f"""
    <div class="section">
        <h3>{fv.scenario_name} ({fv.scenario_type.value})</h3>
        {metrics}
        <div class="grid-2">
            <div class="chart-container">{topdown_html}</div>
            <div class="chart-container">{cache_html}</div>
        </div>
        <div class="chart-container">{instmix_html}</div>
        <h4>Bottleneck Analysis</h4>
        {notes_html}
    </div>"""
