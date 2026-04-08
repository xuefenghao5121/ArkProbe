"""Main report orchestrator — generates comprehensive HTML reports.

Assembles all analysis results into a single self-contained HTML report
with interactive Plotly charts, designed for chip architects.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from ..model.schema import WorkloadFeatureVector
from ..analysis.bottleneck_analyzer import BottleneckAnalyzer, BottleneckReport
from ..analysis.comparator import ComparisonReport, WorkloadComparator
from ..analysis.design_space import DesignSensitivityReport, DesignSpaceExplorer
from ..analysis.optimization_analyzer import (
    OptimizationAnalyzer,
    OptimizationReport as OptReport,
)
from .sections import (
    render_cross_scenario,
    render_design_recommendations,
    render_executive_summary,
    render_optimization_recommendations,
    render_scenario_section,
)

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


class ReportGenerator:
    """Generates comprehensive HTML reports for chip architects."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(".")
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=False,  # We generate safe HTML ourselves
        )

    def generate_full_report(
        self,
        feature_vectors: List[WorkloadFeatureVector],
        title: str = "Workload Characterization Report",
        priority_weights: Optional[Dict[str, float]] = None,
        output_file: Optional[Path] = None,
    ) -> Path:
        """Generate the complete HTML report.

        This is the main entry point. It runs all analysis engines
        and assembles the results into an interactive HTML report.
        """
        if not feature_vectors:
            raise ValueError("No feature vectors provided")

        log.info("Generating report for %d scenarios...", len(feature_vectors))

        # Run analysis engines
        bottleneck_analyzer = BottleneckAnalyzer()
        comparator = WorkloadComparator()
        design_explorer = DesignSpaceExplorer()

        # Bottleneck analysis per scenario
        bottleneck_reports: Dict[str, BottleneckReport] = {}
        for fv in feature_vectors:
            try:
                bottleneck_reports[fv.scenario_name] = bottleneck_analyzer.analyze(fv)
            except Exception as e:
                log.error("Bottleneck analysis failed for %s: %s", fv.scenario_name, e)

        # Cross-scenario comparison
        comparison: Optional[ComparisonReport] = None
        if len(feature_vectors) >= 2:
            try:
                comparison = comparator.compare(feature_vectors)
            except Exception as e:
                log.error("Comparison failed: %s", e)

        # Design space sensitivity
        sensitivity_report = design_explorer.full_analysis(
            feature_vectors, priority_weights
        )

        # Render sections
        executive_html = render_executive_summary(
            feature_vectors, bottleneck_reports, sensitivity_report
        )

        scenario_html = '<div class="section" id="scenario-analysis"><h2>2. Scenario Analysis</h2>'
        for fv in feature_vectors:
            br = bottleneck_reports.get(fv.scenario_name)
            scenario_html += render_scenario_section(fv, br)
        scenario_html += "</div>"

        cross_html = ""
        if comparison:
            cross_html = render_cross_scenario(feature_vectors, comparison)

        design_html = render_design_recommendations(
            feature_vectors, sensitivity_report
        )

        # Platform optimization analysis
        optimization_analyzer = OptimizationAnalyzer()
        optimization_reports: Dict[str, OptReport] = {}
        for fv in feature_vectors:
            try:
                optimization_reports[fv.scenario_name] = optimization_analyzer.analyze(fv)
            except Exception as e:
                log.error("Optimization analysis failed for %s: %s", fv.scenario_name, e)

        cross_optimization = None
        if len(feature_vectors) >= 2:
            try:
                cross_optimization = optimization_analyzer.cross_scenario_analysis(
                    feature_vectors)
            except Exception as e:
                log.error("Cross-scenario optimization failed: %s", e)

        optimization_html = render_optimization_recommendations(
            feature_vectors, optimization_reports, cross_optimization
        )

        # Render final HTML
        template = self.jinja_env.get_template("report_base.html")
        platform = feature_vectors[0].platform if feature_vectors else "Unknown"
        html = template.render(
            title=title,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            platform=platform,
            scenario_count=len(feature_vectors),
            version="0.1.0",
            executive_summary_html=executive_html,
            scenario_sections_html=scenario_html,
            cross_scenario_html=cross_html,
            design_recommendations_html=design_html,
            optimization_recommendations_html=optimization_html,
        )

        # Write output
        if output_file is None:
            output_file = self.output_dir / "report.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding="utf-8")

        log.info("Report generated: %s", output_file)
        return output_file
