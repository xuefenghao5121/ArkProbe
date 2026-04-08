"""Report section renderers for the HTML report."""

from .executive_summary import render_executive_summary
from .scenario_deep_dive import render_scenario_section
from .cross_scenario import render_cross_scenario
from .design_recommendations import render_design_recommendations
from .optimization_recommendations import render_optimization_recommendations

__all__ = [
    "render_executive_summary",
    "render_scenario_section",
    "render_cross_scenario",
    "render_design_recommendations",
    "render_optimization_recommendations",
]
