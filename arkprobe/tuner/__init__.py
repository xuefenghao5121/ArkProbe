"""Hardware tuner module for microarchitectural parameter adjustment.

This module provides capabilities to:
1. Tune real hardware parameters (CPU frequency, SMT, C-states, NUMA)
2. Run workloads under different configurations
3. Compare performance impact across configurations
"""

from .hardware_tuner import HardwareTuner, TuningConfig, TuningResult
from .comparator import TuningComparator, ImpactReport

__all__ = [
    "HardwareTuner",
    "TuningConfig",
    "TuningResult",
    "TuningComparator",
    "ImpactReport",
]
