"""Per-scenario deep dive section renderer."""
from __future__ import annotations
from typing import Optional
from ...model.schema import WorkloadFeatureVector
from ...analysis.bottleneck_analyzer import BottleneckReport
from ..charts import ChartFactory


def _get_severity_class(value: float, thresholds: tuple) -> str:
    """Get severity class based on value and thresholds.

    thresholds: (good, moderate, high) - value below first is good, etc.
    """
    if value < thresholds[0]:
        return "severity-good"
    elif value < thresholds[1]:
        return "severity-moderate"
    elif value < thresholds[2]:
        return "severity-significant"
    else:
        return "severity-critical"


def _render_metric_card(value: float, label: str, unit: str = "",
                        thresholds: tuple = None, format_str: str = ".1f") -> str:
    """Render a metric card with severity coloring."""
    severity_class = ""
    if thresholds:
        severity_class = _get_severity_class(value, thresholds)

    display_value = f"{value:{format_str}}{unit}"
    return f"""
    <div class="metric-card {severity_class}">
        <div class="value">{display_value}</div>
        <div class="label">{label}</div>
    </div>"""


def render_scenario_section(
    fv: WorkloadFeatureVector,
    bottleneck: Optional[BottleneckReport] = None,
) -> str:
    """Render a single scenario deep-dive section.

    Shows:
    1. Key metrics cards with severity indicators
    2. TopDown and cache charts
    3. Bottleneck analysis with clear indicators and recommendations
    """
    # Generate charts
    topdown_fig = ChartFactory.topdown_stacked_bar(fv)
    cache_fig = ChartFactory.cache_mpki_waterfall(fv)
    instmix_fig = ChartFactory.instruction_mix_pie(fv)

    topdown_html = topdown_fig.to_html(include_plotlyjs=False, full_html=False)
    cache_html = cache_fig.to_html(include_plotlyjs=False, full_html=False)
    instmix_html = instmix_fig.to_html(include_plotlyjs=False, full_html=False)

    # Key metrics cards with severity indicators
    # IPC: good > 2, moderate > 1, low < 1
    # L3 MPKI: good < 5, moderate < 15, high < 25
    # Branch MPKI: good < 5, moderate < 10, high < 20
    metrics = f"""
    <div class="grid-4">
        {_render_metric_card(fv.compute.ipc, "IPC", "", thresholds=(2.0, 3.0, 4.0), format_str=".2f")}
        {_render_metric_card(fv.cache.l3_mpki, "L3 MPKI", "", thresholds=(5.0, 15.0, 25.0))}
        {_render_metric_card(fv.branch.branch_mpki, "Branch MPKI", "", thresholds=(5.0, 10.0, 20.0))}
        {_render_metric_card(fv.memory.bandwidth_utilization * 100, "Mem BW", "%", thresholds=(50, 70, 85), format_str=".0f")}
    </div>"""

    # Bottleneck analysis narrative with improved formatting
    notes_html = ""
    if bottleneck:
        # Primary bottleneck summary
        severity = "critical" if bottleneck.primary_score > 0.5 else \
                   "significant" if bottleneck.primary_score > 0.3 else \
                   "moderate" if bottleneck.primary_score > 0.15 else "good"

        notes_html += f"""
        <div class="bottleneck-summary severity-{severity}">
            <h4>Primary Bottleneck: {bottleneck.primary_bottleneck.value.replace('_', ' ').title()}</h4>
            <p>Severity: {bottleneck.primary_score:.0%}</p>
        </div>"""

        # Architect notes
        notes_html += '<div class="architect-notes">'
        for note in bottleneck.architect_notes:
            notes_html += f'<p class="note-item">{note}</p>'
        notes_html += '</div>'

        # Detailed analysis
        for detail in bottleneck.details:
            # Determine severity based on score
            detail_severity = "significant" if detail.score > 0.3 else "moderate"
            notes_html += f"""
            <div class="detail-section severity-{detail_severity}">
                <h5>{detail.category} ({detail.score:.0%})</h5>
                <div class="indicators">
                    <strong>Indicators:</strong>
                    <ul>"""

            for ind in detail.indicators:
                notes_html += f'<li>{ind}</li>'

            notes_html += '</ul></div>'

            if detail.recommendations:
                notes_html += '<div class="recommendations"><strong>Recommendations:</strong><ul>'
                for rec in detail.recommendations:
                    # Highlight critical recommendations
                    is_critical = "CRITICAL" in rec or "URGENT" in rec
                    rec_class = "recommendation-critical" if is_critical else ""
                    notes_html += f'<li class="{rec_class}">{rec}</li>'
                notes_html += '</ul></div>'

            notes_html += '</div>'

    return f"""
    <div class="section">
        <h3>{fv.scenario_name} <span class="scenario-type">({fv.scenario_type.value})</span></h3>
        {metrics}
        <div class="grid-2">
            <div class="chart-container">{topdown_html}</div>
            <div class="chart-container">{cache_html}</div>
        </div>
        <div class="chart-container">{instmix_html}</div>
        <h4>Bottleneck Analysis</h4>
        {notes_html}
    </div>"""
