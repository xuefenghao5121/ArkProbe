"""Plotly chart generation for the HTML report.

All methods return plotly Figure objects that can be serialized
to JSON and embedded in the HTML report.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..model.schema import WorkloadFeatureVector


# Color palette for consistent styling
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]

TOPDOWN_COLORS = {
    "Frontend Bound": "#ff6b6b",
    "Backend Bound": "#4ecdc4",
    "Retiring": "#45b7d1",
    "Bad Speculation": "#f9ca24",
}

# Severity colors for bottleneck indication
SEVERITY_COLORS = {
    "critical": "#d62728",    # Red
    "significant": "#ff7f0e", # Orange
    "moderate": "#f9ca24",    # Yellow
    "good": "#2ca02c",        # Green
}

# Threshold lines for reference
THRESHOLD_LINES = {
    "l3_mpki_high": {"value": 5.0, "label": "High L3 MPKI (5)"},
    "l3_mpki_critical": {"value": 15.0, "label": "Critical L3 MPKI (15)"},
    "ipc_good": {"value": 2.0, "label": "Good IPC (2.0)"},
    "ipc_low": {"value": 1.0, "label": "Low IPC (1.0)"},
}


class ChartFactory:
    """Generate Plotly charts for the HTML report."""

    @staticmethod
    def topdown_stacked_bar(fv: WorkloadFeatureVector) -> go.Figure:
        """TopDown L1 breakdown as a stacked horizontal bar with bottleneck highlight.

        The dominant component is highlighted with a border and annotation.
        """
        td = fv.compute.topdown_l1
        categories = ["Frontend Bound", "Backend Bound", "Retiring", "Bad Speculation"]
        values = [td.frontend_bound, td.backend_bound, td.retiring, td.bad_speculation]

        # Find dominant component (excluding Retiring)
        non_retiring = [(c, v) for c, v in zip(categories, values) if c != "Retiring"]
        dominant = max(non_retiring, key=lambda x: x[1])

        fig = go.Figure()
        for cat, val in zip(categories, values):
            # Highlight dominant bottleneck
            is_dominant = (cat == dominant[0] and val > 0.15)

            fig.add_trace(go.Bar(
                y=[fv.scenario_name],
                x=[val * 100],
                name=cat,
                orientation="h",
                marker_color=TOPDOWN_COLORS[cat],
                marker_line_width=3 if is_dominant else 0,
                marker_line_color="#d62728" if is_dominant else None,
                text=f"{val:.0%}",
                textposition="inside",
            ))

        # Add annotation for dominant bottleneck
        if dominant[1] > 0.15:
            fig.add_annotation(
                x=50, y=0,
                text=f"⚠ Primary: {dominant[0]} ({dominant[1]:.0%})",
                showarrow=False,
                font=dict(size=12, color="#d62728"),
                xanchor="center",
            )

        fig.update_layout(
            barmode="stack",
            title=f"TopDown L1 — {fv.scenario_name}",
            xaxis_title="Percentage (%)",
            xaxis=dict(range=[0, 100]),
            height=200,
            margin=dict(l=20, r=20, t=40, b=30),
            legend=dict(orientation="h", y=-0.3),
        )
        return fig

    @staticmethod
    def topdown_comparison(feature_vectors: List[WorkloadFeatureVector]) -> go.Figure:
        """Compare TopDown L1 across multiple workloads."""
        names = [fv.scenario_name for fv in feature_vectors]

        fig = go.Figure()
        for cat, color in TOPDOWN_COLORS.items():
            key = cat.lower().replace(" ", "_")
            values = [getattr(fv.compute.topdown_l1, key) * 100 for fv in feature_vectors]
            fig.add_trace(go.Bar(
                y=names, x=values, name=cat,
                orientation="h", marker_color=color,
            ))

        fig.update_layout(
            barmode="stack",
            title="TopDown L1 Comparison",
            xaxis_title="Percentage (%)",
            xaxis=dict(range=[0, 100]),
            height=max(250, len(names) * 40),
            margin=dict(l=20, r=20, t=40, b=30),
            legend=dict(orientation="h", y=-0.2),
        )
        return fig

    @staticmethod
    def cache_mpki_waterfall(fv: WorkloadFeatureVector) -> go.Figure:
        """L1I -> L1D -> L2 -> L3 MPKI as a bar chart with severity coloring.

        Colors indicate severity:
        - Green: Good (L3 < 5, L2 < 10)
        - Yellow: Moderate (L3 5-15, L2 10-20)
        - Orange: High (L3 15-25, L2 20-30)
        - Red: Critical (L3 > 25, L2 > 30)
        """
        levels = ["L1I", "L1D", "L2", "L3"]
        values = [fv.cache.l1i_mpki, fv.cache.l1d_mpki, fv.cache.l2_mpki, fv.cache.l3_mpki]

        # Color based on severity
        def get_severity_color(level: str, value: float) -> str:
            if level == "L3":
                if value < 5:
                    return "#2ca02c"  # Green
                elif value < 15:
                    return "#f9ca24"  # Yellow
                elif value < 25:
                    return "#ff7f0e"  # Orange
                else:
                    return "#d62728"  # Red
            elif level == "L2":
                if value < 10:
                    return "#2ca02c"
                elif value < 20:
                    return "#f9ca24"
                elif value < 30:
                    return "#ff7f0e"
                else:
                    return "#d62728"
            elif level == "L1D":
                if value < 20:
                    return "#2ca02c"
                elif value < 40:
                    return "#f9ca24"
                else:
                    return "#ff7f0e"
            else:  # L1I
                if value < 3:
                    return "#2ca02c"
                elif value < 5:
                    return "#f9ca24"
                else:
                    return "#ff7f0e"

        colors = [get_severity_color(l, v) for l, v in zip(levels, values)]

        fig = go.Figure(go.Bar(
            x=levels, y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
        ))

        # Add threshold reference lines
        fig.add_hline(
            y=5.0, line_dash="dash", line_color="#f9ca24", opacity=0.7,
            annotation_text="L3 threshold (5)", annotation_position="right"
        )
        fig.add_hline(
            y=15.0, line_dash="dash", line_color="#d62728", opacity=0.7,
            annotation_text="L3 critical (15)", annotation_position="right"
        )

        fig.update_layout(
            title=f"Cache MPKI — {fv.scenario_name}",
            yaxis_title="MPKI (misses per kilo-instructions)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=30),
        )
        return fig

    @staticmethod
    def instruction_mix_pie(fv: WorkloadFeatureVector) -> go.Figure:
        """Instruction mix as a pie/donut chart."""
        im = fv.compute.instruction_mix
        labels = ["Integer", "FP", "Vector", "Branch", "Load", "Store", "Other"]
        values = [im.integer_ratio, im.fp_ratio, im.vector_ratio,
                  im.branch_ratio, im.load_ratio, im.store_ratio, im.other_ratio]

        # Filter out zero values
        filtered = [(l, v) for l, v in zip(labels, values) if v > 0.001]
        if not filtered:
            filtered = [("No data", 1.0)]

        fig = go.Figure(go.Pie(
            labels=[f[0] for f in filtered],
            values=[f[1] for f in filtered],
            hole=0.4,
            textinfo="label+percent",
        ))

        fig.update_layout(
            title=f"Instruction Mix — {fv.scenario_name}",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    @staticmethod
    def radar_overlay(
        feature_vectors: List[WorkloadFeatureVector],
        radar_data: Dict,
    ) -> go.Figure:
        """Radar chart comparing all workloads on key dimensions."""
        dimensions = radar_data["dimensions"]
        series = radar_data["series"]

        fig = go.Figure()
        for i, s in enumerate(series):
            values = s["values"] + [s["values"][0]]  # close the polygon
            theta = dimensions + [dimensions[0]]
            fig.add_trace(go.Scatterpolar(
                r=values, theta=theta,
                fill="toself", name=s["name"],
                opacity=0.3,
                line=dict(color=COLORS[i % len(COLORS)]),
            ))

        fig.update_layout(
            title="Cross-Scenario Radar Comparison",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    @staticmethod
    def sensitivity_heatmap(matrix: pd.DataFrame) -> go.Figure:
        """Scenarios x Design Parameters heatmap."""
        fig = go.Figure(go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            text=np.round(matrix.values, 2),
            texttemplate="%{text}",
            colorbar_title="Sensitivity",
        ))

        fig.update_layout(
            title="Hardware Design Parameter Sensitivity Matrix",
            xaxis_title="Design Parameter",
            yaxis_title="Workload",
            height=max(400, len(matrix) * 40),
            margin=dict(l=20, r=20, t=60, b=80),
            xaxis=dict(tickangle=-45),
        )
        return fig

    @staticmethod
    def scalability_lines(feature_vectors: List[WorkloadFeatureVector]) -> go.Figure:
        """Multi-line chart: core count vs efficiency per scenario."""
        fig = go.Figure()

        for i, fv in enumerate(feature_vectors):
            if fv.scalability is None:
                continue
            fig.add_trace(go.Scatter(
                x=fv.scalability.core_counts,
                y=fv.scalability.scaling_efficiency,
                mode="lines+markers",
                name=fv.scenario_name,
                line=dict(color=COLORS[i % len(COLORS)]),
            ))

        # Ideal scaling line
        if any(fv.scalability for fv in feature_vectors):
            fig.add_trace(go.Scatter(
                x=[1, 128], y=[1.0, 1.0],
                mode="lines", name="Ideal",
                line=dict(color="gray", dash="dash"),
            ))

        fig.update_layout(
            title="Multi-Core Scaling Efficiency",
            xaxis_title="Core Count",
            yaxis_title="Scaling Efficiency (1.0 = ideal)",
            yaxis=dict(range=[0, 1.2]),
            height=400,
            margin=dict(l=20, r=20, t=40, b=30),
        )
        return fig

    @staticmethod
    def pca_scatter(pca_data: Dict) -> go.Figure:
        """2D PCA scatter plot with workload labels."""
        points = pca_data.get("points", [])
        if not points:
            return go.Figure()

        ev = pca_data.get("explained_variance", [0, 0])

        fig = go.Figure()
        # Color by scenario type
        type_colors = {}
        for p in points:
            if p["type"] not in type_colors:
                type_colors[p["type"]] = COLORS[len(type_colors) % len(COLORS)]

        seen_types = set()
        for p in points:
            fig.add_trace(go.Scatter(
                x=[p["x"]], y=[p["y"]],
                mode="markers+text",
                name=p["type"],
                text=[p["name"]],
                textposition="top center",
                marker=dict(size=12, color=type_colors[p["type"]]),
                showlegend=p["type"] not in seen_types,
            ))
            seen_types.add(p["type"])

        fig.update_layout(
            title="Workload PCA Projection",
            xaxis_title=f"PC1 ({ev[0]:.0%} variance)" if ev else "PC1",
            yaxis_title=f"PC2 ({ev[1]:.0%} variance)" if len(ev) > 1 else "PC2",
            height=500,
            margin=dict(l=20, r=20, t=40, b=30),
        )
        return fig

    @staticmethod
    def bandwidth_latency_scatter(
        feature_vectors: List[WorkloadFeatureVector],
    ) -> go.Figure:
        """Memory bandwidth vs latency scatter, bubble size = L3 MPKI."""
        fig = go.Figure()

        for i, fv in enumerate(feature_vectors):
            bw = fv.memory.bandwidth_read_gbps + fv.memory.bandwidth_write_gbps
            lat = fv.memory.avg_latency_ns if fv.memory.avg_latency_ns is not None else 50
            size = max(fv.cache.l3_mpki * 3, 8)

            fig.add_trace(go.Scatter(
                x=[bw], y=[lat],
                mode="markers+text",
                text=[fv.scenario_name],
                textposition="top center",
                marker=dict(size=size, color=COLORS[i % len(COLORS)], opacity=0.7),
                name=fv.scenario_name,
            ))

        fig.update_layout(
            title="Memory Bandwidth vs Latency (bubble size = L3 MPKI)",
            xaxis_title="Bandwidth (GB/s)",
            yaxis_title="Latency (ns)",
            height=450,
            margin=dict(l=20, r=20, t=40, b=30),
        )
        return fig

    @staticmethod
    def recommendation_bar(recommendations: list) -> go.Figure:
        """Ranked design recommendations as horizontal bar chart."""
        if not recommendations:
            return go.Figure()

        params = [r.parameter for r in recommendations[:10]]
        priorities = [r.priority for r in recommendations[:10]]
        costs = [r.area_cost for r in recommendations[:10]]

        cost_colors = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}
        colors = [cost_colors.get(c, "#7f7f7f") for c in costs]

        fig = go.Figure(go.Bar(
            y=params, x=priorities,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.2f}" for p in priorities],
            textposition="outside",
        ))

        fig.update_layout(
            title="Design Recommendations (cost-adjusted priority)",
            xaxis_title="Priority Score",
            height=max(300, len(params) * 35),
            margin=dict(l=20, r=20, t=40, b=30),
        )
        return fig

    @staticmethod
    def optimization_gap_heatmap(matrix: pd.DataFrame) -> go.Figure:
        """Scenarios x Tuning Parameters impact heatmap for optimization."""
        if matrix.empty:
            return go.Figure()

        fig = go.Figure(go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=[
                [0, "#1a472a"],     # low impact — green (already tuned)
                [0.3, "#2d6a4f"],
                [0.5, "#f4a261"],   # medium impact — orange
                [0.7, "#e76f51"],
                [1.0, "#d62728"],   # high impact — red (needs tuning)
            ],
            text=matrix.map(lambda v: f"{v:.2f}").values,
            texttemplate="%{text}",
            colorbar_title="Impact",
        ))

        fig.update_layout(
            title="Platform Optimization Impact Matrix (scenarios x parameters)",
            xaxis_title="Tuning Parameter",
            yaxis_title="Scenario",
            height=max(350, len(matrix) * 40 + 100),
            margin=dict(l=20, r=20, t=40, b=120),
            xaxis_tickangle=-45,
        )
        return fig

    @staticmethod
    def optimization_score_bars(scores: Dict[str, float]) -> go.Figure:
        """Per-scenario optimization score as horizontal bars."""
        if not scores:
            return go.Figure()

        names = list(scores.keys())
        values = list(scores.values())

        colors = []
        for v in values:
            if v >= 80:
                colors.append("#2ca02c")
            elif v >= 50:
                colors.append("#ff7f0e")
            else:
                colors.append("#d62728")

        fig = go.Figure(go.Bar(
            y=names, x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}/100" for v in values],
            textposition="outside",
        ))

        fig.update_layout(
            title="Platform Optimization Score (higher = better tuned)",
            xaxis_title="Score",
            xaxis_range=[0, 110],
            height=max(250, len(names) * 40),
            margin=dict(l=20, r=20, t=40, b=30),
        )
        return fig
