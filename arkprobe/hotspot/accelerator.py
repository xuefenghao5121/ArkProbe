"""
End-to-end hotspot acceleration pipeline.

Orchestrates: JFR profiling → pattern classification → C++ generation → compilation → benchmarking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .profiler import JfrHotspotProfiler, BytecodeExtractor
from .analyzer import PatternMatcher, PatternClassification, classify_hotspot_method
from .codegen import CppGenerator, GenerationConfig
from .compiler import Compiler, CompileResult
from .runtime import BenchmarkRunner, JNILoader

log = logging.getLogger(__name__)


@dataclass
class AccelerationConfig:
    """Configuration for end-to-end acceleration pipeline."""
    output_dir: Path
    library_name: str = "arkprobe_hotspot"
    min_cpu_percent: float = 2.0
    min_speedup: float = 1.5
    optimization_level: str = "O3"
    target_arch: Optional[str] = None


@dataclass
class AccelerationResult:
    """Result of the end-to-end acceleration pipeline."""
    methods_analyzed: int = 0
    methods_classified: int = 0
    methods_accelerated: int = 0
    generated_files: list[Path] = field(default_factory=list)
    compiled_libraries: list[Path] = field(default_factory=list)
    benchmark_results: list[dict] = field(default_factory=list)
    recommended_methods: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "methods_analyzed": self.methods_analyzed,
            "methods_classified": self.methods_classified,
            "methods_accelerated": self.methods_accelerated,
            "generated_files": [str(f) for f in self.generated_files],
            "compiled_libraries": [str(f) for f in self.compiled_libraries],
            "benchmark_results": self.benchmark_results,
            "recommended_methods": self.recommended_methods,
        }


class HotspotAccelerator:
    """End-to-end hotspot detection and C++ acceleration pipeline."""

    def __init__(
        self,
        jvm_pid: int,
        config: Optional[AccelerationConfig] = None,
    ):
        self.jvm_pid = jvm_pid
        self.config = config or AccelerationConfig(
            output_dir=Path("./hotspot_output")
        )
        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.profiler = JfrHotspotProfiler(output_dir=self.config.output_dir / "profiler")
        self.bytecode_extractor = BytecodeExtractor(output_dir=self.config.output_dir / "bytecode")
        self.pattern_matcher = PatternMatcher()
        # Resolve target_arch: None means auto-detect
        resolved_arch = self.config.target_arch
        if resolved_arch is None:
            from .compiler.compiler import detect_host_arch
            resolved_arch = detect_host_arch()
            log.info("Auto-detected target architecture: %s", resolved_arch)

        self.codegen = CppGenerator(GenerationConfig(
            output_dir=self.config.output_dir / "codegen",
            library_name=self.config.library_name,
            optimization_level=self.config.optimization_level,
            target_arch=resolved_arch,
        ))
        self.compiler = Compiler(
            build_dir=self.config.output_dir / "build",
            target_arch=resolved_arch,
        )

    def run(
        self,
        profiling_duration: int = 30,
        run_benchmark: bool = True,
    ) -> AccelerationResult:
        """Run the full hotspot acceleration pipeline."""
        result = AccelerationResult()

        # Step 1: JFR profiling
        log.info("Step 1: JFR profiling (%ds)...", profiling_duration)
        profile = self.profiler.profile(self.jvm_pid, profiling_duration)
        result.methods_analyzed = len(profile.methods)

        if not profile.methods:
            log.warning("No hotspot methods found in JFR profile")
            return result

        # Step 2: Pattern classification
        log.info("Step 2: Pattern classification...")
        classified_methods = []
        for method in profile.methods:
            classification = self.pattern_matcher.classify(method)

            # Update method with classification results
            method.pattern_type = classification.pattern_type
            method.simd_potential = self.pattern_matcher.estimate_simd_potential(method)
            method.deopt_risk = self.pattern_matcher.estimate_deopt_risk(method)

            # Skip unknown or low-confidence classifications
            if classification.pattern_type == "unknown" or classification.confidence < 0.5:
                continue

            classified_methods.append((method, classification))
            result.methods_classified += 1

        log.info("Classified %d methods for acceleration", len(classified_methods))

        # Step 3: C++ code generation
        log.info("Step 3: C++ code generation...")
        generated = self._generate_code(classified_methods)
        result.generated_files.extend(generated)

        # Step 4: Compilation
        log.info("Step 4: Compiling to shared library...")
        compile_result = self.compiler.compile(self.codegen)
        if compile_result.success:
            result.compiled_libraries.append(compile_result.output_file)
            result.methods_accelerated = len(generated)
        else:
            log.error("Compilation failed: %s", compile_result.error)

        # Step 5: Benchmarking (optional)
        if run_benchmark and compile_result.success:
            log.info("Step 5: Running benchmarks...")
            benchmarks = self._run_benchmarks(classified_methods, compile_result.output_file)
            result.benchmark_results = [b.to_dict() for b in benchmarks]

            for bmr in benchmarks:
                if bmr.is_worth_accelerating(self.config.min_speedup):
                    result.recommended_methods.append({
                        "method": bmr.method_name,
                        "speedup": bmr.speedup_factor,
                        "java_time_ms": bmr.java_time_ms,
                        "cpp_time_ms": bmr.cpp_time_ms,
                    })

        # Save result
        self._save_result(result)

        return result

    def _generate_code(
        self,
        methods: list[tuple],
    ) -> list[Path]:
        """Generate C++ code for classified methods."""
        generated = []
        method_cpp_pairs = []

        for method, classification in methods:
            bytecode = self.bytecode_extractor.get_method_bytecode(
                self.jvm_pid,
                method.name.rsplit(".", 1)[0],
                method.name.rsplit(".", 1)[-1],
            )

            output_file = self.codegen.generate_for_method(
                method, classification, bytecode
            )
            generated.append(output_file)
            method_cpp_pairs.append((method, classification))

        # Generate JNI bridge
        bridge_file = self.codegen.generate_jni_bridge(method_cpp_pairs)
        generated.append(bridge_file)

        # Generate CMakeLists
        cmake_file = self.codegen.generate_cmake(generated)
        generated.append(cmake_file)

        return generated

    def _run_benchmarks(
        self,
        methods: list[tuple],
        so_path: Optional[Path],
    ) -> list:
        """Run benchmarks for generated methods."""
        if so_path is None:
            return []

        runner = BenchmarkRunner(self.jvm_pid)
        results = []

        for method, _ in methods:
            try:
                benchmark = runner.benchmark_method(
                    method, so_path, iterations=100, warmup_iters=10
                )
                results.append(benchmark)
            except Exception as e:
                log.warning("Benchmark failed for %s: %s", method.name, e)

        return results

    def _save_result(self, result: AccelerationResult) -> None:
        """Save acceleration result to JSON."""
        output_file = self.config.output_dir / "acceleration_result.json"
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        log.info("Result saved to %s", output_file)

    def generate_report(self, result: AccelerationResult) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            "Hotspot Acceleration Report",
            "=" * 60,
            "",
            f"Methods analyzed: {result.methods_analyzed}",
            f"Methods classified: {result.methods_classified}",
            f"Methods accelerated: {result.methods_accelerated}",
            "",
        ]

        if result.recommended_methods:
            lines.extend([
                "Recommended Methods for Production:",
                "-" * 40,
            ])
            for rec in result.recommended_methods:
                lines.append(
                    f"  • {rec['method']}: {rec['speedup']:.2f}x speedup "
                    f"({rec['java_time_ms']:.2f}ms → {rec['cpp_time_ms']:.2f}ms)"
                )
        else:
            lines.append("No methods meet the speedup threshold.")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)


def accelerate(
    jvm_pid: int,
    output_dir: str | Path = "./hotspot_output",
    profiling_duration: int = 30,
    run_benchmark: bool = True,
) -> AccelerationResult:
    """Convenience function to run the full acceleration pipeline."""
    config = AccelerationConfig(output_dir=Path(output_dir))
    accelerator = HotspotAccelerator(jvm_pid, config)
    return accelerator.run(profiling_duration, run_benchmark)
