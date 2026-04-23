"""
Hotspot module: JVM hotspot detection and C++ acceleration.

Integrates:
- JFR-based hotspot profiling (ExecutionSample, Compilation events)
- Pattern matching (vector expressions, string ops, math)
- C++ code generation via Jinja2 templates
- CMake-based compilation to JNI .so
- Runtime JNI loading and benchmarking
"""

from __future__ import annotations

from typing import Optional

from .models import HotspotMethod, HotspotProfile
from .profiler import JfrHotspotProfiler
from .analyzer import PatternMatcher, PatternClassification, classify_hotspot_method
from .codegen import CppGenerator, GenerationConfig, generate_cpp_code
from .compiler import Compiler, CompileResult, compile_hotspot
from .runtime import JNILoader, BenchmarkRunner, load_and_benchmark
from .accelerator import (
    HotspotAccelerator,
    AccelerationConfig,
    AccelerationResult,
    accelerate,
)

__all__ = [
    "HotspotMethod",
    "HotspotProfile",
    "JfrHotspotProfiler",
    "PatternMatcher",
    "PatternClassification",
    "classify_hotspot_method",
    "CppGenerator",
    "GenerationConfig",
    "generate_cpp_code",
    "Compiler",
    "CompileResult",
    "compile_hotspot",
    "JNILoader",
    "BenchmarkRunner",
    "load_and_benchmark",
    "HotspotAccelerator",
    "AccelerationConfig",
    "AccelerationResult",
    "accelerate",
]
