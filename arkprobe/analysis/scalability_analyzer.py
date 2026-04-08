"""Multi-core scalability analysis with Amdahl's law fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from ..model.schema import ScalabilityProfile

log = logging.getLogger(__name__)


class ScalabilityAnalyzer:
    """Analyze multi-core scaling behavior."""

    def analyze(
        self,
        core_counts: List[int],
        throughputs: List[float],
    ) -> ScalabilityProfile:
        """Compute scaling efficiency and fit Amdahl's law."""
        if len(core_counts) != len(throughputs) or len(core_counts) < 2:
            raise ValueError("Need at least 2 data points with matching lengths")

        efficiencies = self._compute_efficiency(core_counts, throughputs)
        serial_fraction = self._fit_amdahl(core_counts, throughputs)
        optimal_cores = self._find_optimal_cores(core_counts, throughputs)

        return ScalabilityProfile(
            core_counts=core_counts,
            throughput_at_core_count=throughputs,
            scaling_efficiency=efficiencies,
            optimal_core_count=optimal_cores,
            amdahl_serial_fraction=serial_fraction,
        )

    def _compute_efficiency(
        self, core_counts: List[int], throughputs: List[float]
    ) -> List[float]:
        """Compute scaling efficiency at each core count.

        efficiency = (throughput[i] / throughput[0]) / (core_counts[i] / core_counts[0])
        Perfect linear scaling = 1.0
        """
        base_tp = throughputs[0]
        base_cores = core_counts[0]
        if base_tp <= 0:
            return [0.0] * len(core_counts)

        efficiencies = []
        for i in range(len(core_counts)):
            speedup = throughputs[i] / base_tp
            ideal_speedup = core_counts[i] / base_cores
            eff = speedup / ideal_speedup if ideal_speedup > 0 else 0.0
            efficiencies.append(round(min(eff, 2.0), 4))  # cap at 2x for super-linear

        return efficiencies

    def _fit_amdahl(
        self, core_counts: List[int], throughputs: List[float]
    ) -> Optional[float]:
        """Fit Amdahl's law to estimate serial fraction.

        Amdahl's law: speedup(N) = 1 / (s + (1-s)/N)
        where s = serial fraction, N = number of cores

        Equivalently: throughput(N) = T1 * N / (1 + s*(N-1))
        """
        if len(core_counts) < 3:
            return None

        base_tp = throughputs[0]
        base_cores = core_counts[0]
        if base_tp <= 0:
            return None

        # Normalize to speedups relative to single-core
        speedups = [tp / base_tp for tp in throughputs]
        cores_arr = np.array(core_counts, dtype=float)
        speedups_arr = np.array(speedups, dtype=float)

        def amdahl(n, s):
            return 1.0 / (s + (1.0 - s) / n)

        try:
            popt, _ = curve_fit(
                amdahl, cores_arr, speedups_arr / cores_arr[0],
                p0=[0.1], bounds=(0.0, 1.0),
                maxfev=10000,
            )
            return round(float(popt[0]), 4)
        except Exception as e:
            log.warning("Amdahl fit failed: %s", e)
            return None

    def _find_optimal_cores(
        self,
        core_counts: List[int],
        throughputs: List[float],
        threshold: float = 0.10,
    ) -> Optional[int]:
        """Find the core count beyond which marginal gain < threshold.

        Returns the last core count where adding more cores gives > threshold
        improvement in throughput.
        """
        if len(core_counts) < 2:
            return core_counts[0] if core_counts else None

        for i in range(1, len(core_counts)):
            if throughputs[i - 1] <= 0:
                continue
            marginal_gain = (throughputs[i] - throughputs[i - 1]) / throughputs[i - 1]
            if marginal_gain < threshold:
                return core_counts[i - 1]

        return core_counts[-1]  # Still scaling at max tested
