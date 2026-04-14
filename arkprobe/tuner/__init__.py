"""Hardware tuner module for microarchitectural parameter adjustment.

This module provides capabilities to:
1. Tune real hardware parameters (CPU frequency, SMT, C-states, NUMA)
2. Run workloads under different configurations
3. Compare performance impact across configurations
4. Simulate microarchitectural changes with gem5
"""

from .hardware_tuner import HardwareTuner, TuningConfig, TuningResult
from .comparator import TuningComparator, ImpactReport
from .gem5_tuner import Gem5Tuner, Gem5Config, Gem5Stats, O3CPUConfig, CacheConfig, GEM5_PRESETS

__all__ = [
    # Hardware tuning
    "HardwareTuner",
    "TuningConfig",
    "TuningResult",
    # Comparison
    "TuningComparator",
    "ImpactReport",
    # gem5 simulation
    "Gem5Tuner",
    "Gem5Config",
    "Gem5Stats",
    "O3CPUConfig",
    "CacheConfig",
    "GEM5_PRESETS",
]
