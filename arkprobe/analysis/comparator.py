"""Cross-scenario workload comparison with visualization data preparation.

Provides radar charts, heatmaps, PCA scatter plots, and clustering
for comparing diverse workloads on a unified basis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..model.schema import WorkloadFeatureVector

log = logging.getLogger(__name__)


# Default dimensions for radar chart comparison
DEFAULT_RADAR_DIMENSIONS = [
    "ipc", "l3_mpki", "branch_mpki", "bandwidth_util",
    "io_intensity", "network_intensity", "lock_contention", "simd_util",
]


@dataclass
class ComparisonReport:
    """Cross-scenario comparison results."""
    scenario_names: List[str]
    radar_data: Dict  # For radar chart
    heatmap_df: Optional[pd.DataFrame] = None  # Normalized metrics matrix
    clusters: Optional[Dict[int, List[str]]] = None
    pca_data: Optional[Dict] = None


class WorkloadComparator:
    """Cross-scenario comparison with visualization data preparation."""

    def compare(
        self, feature_vectors: List[WorkloadFeatureVector]
    ) -> ComparisonReport:
        """Generate a full cross-scenario comparison."""
        names = [fv.scenario_name for fv in feature_vectors]

        radar = self.radar_chart_data(feature_vectors)
        heatmap = self.heatmap_data(feature_vectors)
        clusters = self.cluster_workloads(feature_vectors)
        pca = self.pca_visualization_data(feature_vectors)

        return ComparisonReport(
            scenario_names=names,
            radar_data=radar,
            heatmap_df=heatmap,
            clusters=clusters,
            pca_data=pca,
        )

    def radar_chart_data(
        self,
        feature_vectors: List[WorkloadFeatureVector],
        dimensions: Optional[List[str]] = None,
    ) -> Dict:
        """Prepare data for radar/spider chart comparison.

        Returns: {
            "dimensions": [...],
            "series": [{"name": "...", "values": [...normalized 0-1...]}, ...]
        }
        """
        if dimensions is None:
            dimensions = DEFAULT_RADAR_DIMENSIONS

        # Extract raw values for each dimension
        raw_data = {}
        for dim in dimensions:
            values = [self._extract_dimension(fv, dim) for fv in feature_vectors]
            raw_data[dim] = values

        # Normalize each dimension to 0-1 (min-max)
        normalized = {}
        for dim, values in raw_data.items():
            vmin = min(values)
            vmax = max(values)
            rng = vmax - vmin
            if rng > 0:
                normalized[dim] = [(v - vmin) / rng for v in values]
            else:
                normalized[dim] = [0.5] * len(values)

        series = []
        for i, fv in enumerate(feature_vectors):
            series.append({
                "name": fv.scenario_name,
                "values": [normalized[dim][i] for dim in dimensions],
            })

        return {"dimensions": dimensions, "series": series}

    def heatmap_data(
        self, feature_vectors: List[WorkloadFeatureVector]
    ) -> pd.DataFrame:
        """Generate normalized heatmap matrix: scenarios x metrics."""
        metrics = [
            "ipc", "l1d_mpki", "l2_mpki", "l3_mpki",
            "branch_mpki", "mispredict_rate",
            "bandwidth_util", "tlb_mpki",
            "io_iops", "net_pps",
            "frontend_bound", "backend_bound", "retiring", "bad_speculation",
            "simd_util", "lock_contention",
        ]

        data = {}
        for fv in feature_vectors:
            row = {}
            for metric in metrics:
                row[metric] = self._extract_dimension(fv, metric)
            data[fv.scenario_name] = row

        df = pd.DataFrame(data).T
        # Normalize columns to 0-1
        for col in df.columns:
            vmin = df[col].min()
            vmax = df[col].max()
            rng = vmax - vmin
            if rng > 0:
                df[col] = (df[col] - vmin) / rng
            else:
                df[col] = 0.5

        return df

    def cluster_workloads(
        self,
        feature_vectors: List[WorkloadFeatureVector],
        n_clusters: int = 4,
    ) -> Dict[int, List[str]]:
        """Cluster workloads by micro-architectural similarity."""
        if len(feature_vectors) < n_clusters:
            n_clusters = max(1, len(feature_vectors) // 2)

        # Build feature matrix
        X = self._build_feature_matrix(feature_vectors)

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Drop zero-variance columns to avoid numerical issues
            col_std = X_scaled.std(axis=0)
            nonzero_mask = col_std > 1e-10
            if nonzero_mask.sum() == 0:
                log.warning("All features have zero variance, skipping clustering")
                return {0: [fv.scenario_name for fv in feature_vectors]}
            X_filtered = X_scaled[:, nonzero_mask]

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_filtered)

            clusters: Dict[int, List[str]] = {}
            for i, label in enumerate(labels):
                clusters.setdefault(int(label), []).append(
                    feature_vectors[i].scenario_name
                )
            return clusters
        except ImportError:
            log.warning("scikit-learn not available, skipping clustering")
            return {0: [fv.scenario_name for fv in feature_vectors]}

    def pca_visualization_data(
        self, feature_vectors: List[WorkloadFeatureVector]
    ) -> Dict:
        """Reduce feature vectors to 2D via PCA for scatter plot."""
        if len(feature_vectors) < 2:
            return {"points": [], "explained_variance": [], "components": []}

        X = self._build_feature_matrix(feature_vectors)

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Drop zero-variance columns after scaling to avoid PCA divide-by-zero
            col_std = X_scaled.std(axis=0)
            nonzero_mask = col_std > 1e-10
            if nonzero_mask.sum() == 0:
                log.warning("All features have zero variance, skipping PCA")
                return {"points": [], "explained_variance": [], "components": []}
            X_filtered = X_scaled[:, nonzero_mask]

            pca = PCA(n_components=min(2, X_filtered.shape[1]))
            X_2d = pca.fit_transform(X_filtered)

            points = []
            for i, fv in enumerate(feature_vectors):
                points.append({
                    "name": fv.scenario_name,
                    "type": fv.scenario_type.value,
                    "x": float(X_2d[i, 0]),
                    "y": float(X_2d[i, 1]) if X_2d.shape[1] > 1 else 0.0,
                })

            return {
                "points": points,
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "components": pca.components_.tolist(),
            }
        except ImportError:
            log.warning("scikit-learn not available, skipping PCA")
            return {"points": [], "explained_variance": [], "components": []}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _extract_dimension(self, fv: WorkloadFeatureVector, dim: str) -> float:
        """Extract a single numeric dimension from a feature vector."""
        mapping = {
            "ipc": fv.compute.ipc,
            "cpi": fv.compute.cpi,
            "l1i_mpki": fv.cache.l1i_mpki,
            "l1d_mpki": fv.cache.l1d_mpki,
            "l2_mpki": fv.cache.l2_mpki,
            "l3_mpki": fv.cache.l3_mpki,
            "branch_mpki": fv.branch.branch_mpki,
            "mispredict_rate": fv.branch.branch_mispredict_rate,
            "bandwidth_util": fv.memory.bandwidth_utilization,
            "tlb_mpki": fv.memory.tlb_mpki or 0.0,
            "io_intensity": fv.io.iops_read + fv.io.iops_write,
            "io_iops": fv.io.iops_read + fv.io.iops_write,
            "network_intensity": fv.network.packets_per_sec_rx + fv.network.packets_per_sec_tx,
            "net_pps": fv.network.packets_per_sec_rx + fv.network.packets_per_sec_tx,
            "lock_contention": fv.concurrency.lock_contention_pct or 0.0,
            "simd_util": fv.compute.simd_utilization,
            "frontend_bound": fv.compute.topdown_l1.frontend_bound,
            "backend_bound": fv.compute.topdown_l1.backend_bound,
            "retiring": fv.compute.topdown_l1.retiring,
            "bad_speculation": fv.compute.topdown_l1.bad_speculation,
        }
        return mapping.get(dim, 0.0)

    def _build_feature_matrix(
        self, feature_vectors: List[WorkloadFeatureVector]
    ) -> np.ndarray:
        """Build a numeric feature matrix for ML operations."""
        features = [
            "ipc", "l1d_mpki", "l2_mpki", "l3_mpki",
            "branch_mpki", "bandwidth_util", "simd_util",
            "frontend_bound", "backend_bound", "retiring",
            "io_intensity", "network_intensity",
        ]

        X = np.zeros((len(feature_vectors), len(features)))
        for i, fv in enumerate(feature_vectors):
            for j, feat in enumerate(features):
                X[i, j] = self._extract_dimension(fv, feat)

        return X
