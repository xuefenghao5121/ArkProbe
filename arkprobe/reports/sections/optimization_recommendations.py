"""Platform optimization recommendations section renderer."""
from __future__ import annotations

from typing import Dict, List, Optional

from ...model.schema import WorkloadFeatureVector
from ...analysis.optimization_analyzer import (
    CrossScenarioOptimizationReport,
    OptimizationReport,
)
from ..charts import ChartFactory


def render_optimization_recommendations(
    feature_vectors: List[WorkloadFeatureVector],
    optimization_reports: Dict[str, OptimizationReport],
    cross_report: Optional[CrossScenarioOptimizationReport] = None,
) -> str:
    """Render Section 5: Platform Optimization Recommendations."""

    # --- Optimization score cards ---
    score_data = {
        name: report.optimization_score
        for name, report in optimization_reports.items()
    }
    score_fig = ChartFactory.optimization_score_bars(score_data)
    score_html = score_fig.to_html(include_plotlyjs=False, full_html=False)

    # --- Per-scenario gap analysis tables ---
    scenario_tables = ""
    for fv in feature_vectors:
        report = optimization_reports.get(fv.scenario_name)
        if not report:
            continue

        scenario_tables += f'<h3>{fv.scenario_name} <span class="tag tag-{_score_tag(report.optimization_score)}">{report.optimization_score:.0f}/100</span></h3>'

        for layer_name in ("os", "bios", "driver"):
            layer = report.layers.get(layer_name)
            if not layer or not layer.recommendations:
                continue

            scenario_tables += f'<h4>{layer_name.upper()} ({layer.gaps_found} gaps / {layer.total_parameters} params)</h4>'
            scenario_tables += """<table>
                <thead><tr>
                    <th>Parameter</th><th>Current</th><th>Recommended</th>
                    <th>Impact</th><th>Difficulty</th><th>Command</th>
                </tr></thead><tbody>"""

            for rec in layer.recommendations:
                gap_class = ""
                if rec.gap_detected is True:
                    gap_class = ' style="border-left: 3px solid #d62728;"'
                elif rec.gap_detected is False:
                    gap_class = ' style="border-left: 3px solid #2ca02c;"'

                impact_tag = _impact_tag(rec.impact_score)
                cmd = rec.apply_commands[0] if rec.apply_commands else "-"

                scenario_tables += f"""<tr{gap_class}>
                    <td><strong>{rec.display_name}</strong><br>
                        <small style="color:var(--text-muted);">{rec.description[:80]}</small></td>
                    <td><code>{rec.current_value}</code></td>
                    <td><code>{rec.recommended_value}</code></td>
                    <td><span class="tag tag-{impact_tag}">{rec.impact_score:.0%}</span></td>
                    <td><span class="tag tag-{rec.difficulty.value}">{rec.difficulty.value}</span></td>
                    <td><code style="font-size:0.8em;">{_escape_html(cmd)}</code></td>
                </tr>"""

            scenario_tables += "</tbody></table>"

    # --- Cross-scenario benefit matrix ---
    matrix_html = ""
    if cross_report and cross_report.parameter_benefit_matrix is not None:
        matrix_fig = ChartFactory.optimization_gap_heatmap(
            cross_report.parameter_benefit_matrix)
        matrix_html = matrix_fig.to_html(include_plotlyjs=False, full_html=False)

    # --- Universal recommendations ---
    universal_html = ""
    if cross_report and cross_report.universal_recommendations:
        universal_html = """<h3>Universal Recommendations (benefit all workloads)</h3>
        <table><thead><tr>
            <th>#</th><th>Parameter</th><th>Value</th>
            <th>Impact</th><th>Difficulty</th><th>Command</th>
        </tr></thead><tbody>"""
        for i, rec in enumerate(cross_report.universal_recommendations[:10]):
            cmd = rec.apply_commands[0] if rec.apply_commands else "-"
            universal_html += f"""<tr>
                <td>{i+1}</td>
                <td><strong>{rec.display_name}</strong></td>
                <td><code>{rec.recommended_value}</code></td>
                <td><span class="tag tag-{_impact_tag(rec.impact_score)}">{rec.impact_score:.0%}</span></td>
                <td>{rec.difficulty.value}</td>
                <td><code>{_escape_html(cmd)}</code></td>
            </tr>"""
        universal_html += "</tbody></table>"

    # --- Conflicting parameters ---
    conflict_html = ""
    if cross_report and cross_report.conflicting_parameters:
        conflict_html = """<h3>Conflicting Parameters (scenarios disagree)</h3>
        <div class="note"><p>These parameters have different optimal values
        depending on workload. Choose based on your primary scenario.</p></div>
        <table><thead><tr>
            <th>Parameter</th><th>Scenario</th><th>Recommended</th>
        </tr></thead><tbody>"""
        for conflict in cross_report.conflicting_parameters:
            scenarios = conflict["scenarios_disagree"]
            first = True
            for scenario, value in scenarios.items():
                name_cell = f'<td rowspan="{len(scenarios)}"><strong>{conflict["parameter"]}</strong></td>' if first else ""
                conflict_html += f"""<tr>
                    {name_cell}
                    <td>{scenario}</td>
                    <td><code>{value}</code></td>
                </tr>"""
                first = False
        conflict_html += "</tbody></table>"

    # --- Actionable script ---
    script_lines = ["#!/bin/bash", "# ArkProbe Platform Optimization Script",
                    "# Generated for maximum cross-scenario benefit", ""]
    if cross_report:
        for rec in cross_report.universal_recommendations:
            if rec.apply_commands and not rec.apply_commands[0].startswith("#"):
                script_lines.append(f"# {rec.display_name}")
                script_lines.extend(rec.apply_commands)
                script_lines.append("")
    elif optimization_reports:
        # Single scenario
        for report in optimization_reports.values():
            for rec in report.all_recommendations:
                if rec.gap_detected and rec.apply_commands and \
                   not rec.apply_commands[0].startswith("#"):
                    script_lines.append(f"# {rec.display_name}")
                    script_lines.extend(rec.apply_commands)
                    script_lines.append("")

    script_content = "\n".join(script_lines)

    return f"""
    <div class="section" id="optimization-recommendations">
        <h2>5. Platform Optimization Recommendations</h2>
        <p>Gap analysis between current platform configuration and recommended settings
        for each workload scenario. Impact scores indicate expected performance benefit
        from applying the recommendation.</p>

        <h3>Optimization Scores</h3>
        <div class="chart-container">{score_html}</div>

        <h3>Per-Scenario Gap Analysis</h3>
        {scenario_tables}

        {"<h3>Cross-Scenario Impact Matrix</h3>" if matrix_html else ""}
        {"<div class='chart-container'>" + matrix_html + "</div>" if matrix_html else ""}

        {universal_html}
        {conflict_html}

        <h3>Optimization Script</h3>
        <p>Copy and review before executing. BIOS-level changes require manual configuration.</p>
        <div class="script-block"><pre>{_escape_html(script_content)}</pre></div>
    </div>"""


def _score_tag(score: float) -> str:
    if score >= 80:
        return "low"    # green
    elif score >= 50:
        return "medium"  # orange
    return "high"       # red


def _impact_tag(impact: float) -> str:
    if impact >= 0.6:
        return "high"    # red
    elif impact >= 0.3:
        return "medium"  # orange
    return "low"        # green


def _escape_html(text: str) -> str:
    import html
    return html.escape(text, quote=True)
