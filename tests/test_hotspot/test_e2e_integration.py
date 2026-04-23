"""
End-to-end integration tests for the hotspot acceleration pipeline.

Tests the full pipeline: profiling → classification → code generation → compilation.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from arkprobe.hotspot.accelerator import (
    HotspotAccelerator,
    AccelerationConfig,
    AccelerationResult,
    accelerate,
)
from arkprobe.hotspot.analyzer import PatternMatcher, PatternClassification
from arkprobe.hotspot.codegen import CppGenerator, GenerationConfig
from arkprobe.hotspot.compiler import Compiler
from arkprobe.hotspot.models import HotspotMethod, HotspotProfile
from arkprobe.hotspot.runtime import BenchmarkRunner


@pytest.fixture
def sample_hotspot_methods():
    """Create sample hotspot methods for testing."""
    return [
        HotspotMethod(
            name="com.example.FastMath.compute",
            signature="(II)D",
            cpu_time_ns=15_200_000,
            cpu_time_percent=5.0,
            bytecode_size=42,
            compilation_count=3,
            inline_count=10,
        ),
        HotspotMethod(
            name="com.example.StringOps.processText",
            signature="(Ljava/lang/String;)Ljava/lang/String;",
            cpu_time_ns=12_800_000,
            cpu_time_percent=4.0,
            bytecode_size=68,
            compilation_count=2,
            inline_count=5,
        ),
        HotspotMethod(
            name="com.example.MathUtils.sigmoid",
            signature="(D)D",
            cpu_time_ns=9_500_000,
            cpu_time_percent=3.0,
            bytecode_size=25,
            compilation_count=1,
            inline_count=0,
        ),
    ]


@pytest.fixture
def sample_profile(sample_hotspot_methods):
    """Create a sample hotspot profile."""
    return HotspotProfile(
        pid=12345,
        jdk_version="openjdk 17.0.9",
        duration_sec=30,
        methods=sample_hotspot_methods,
        total_cpu_time_ns=37_500_000,
    )


class TestPatternMatcherIntegration:
    """Integration tests for pattern matching."""

    def test_classify_multiple_methods(self, sample_hotspot_methods):
        """Test pattern classification on multiple methods."""
        matcher = PatternMatcher()
        classifications = []

        for method in sample_hotspot_methods:
            classification = matcher.classify(method)
            classifications.append(classification)
            assert classification.confidence > 0.0
            assert classification.pattern_type in [
                "vector_expr", "string", "math", "memory_bandwidth", "unknown"
            ]

        # Verify some classifications were made
        assert len(classifications) == len(sample_hotspot_methods)


class TestCodegenIntegration:
    """Integration tests for code generation."""

    def test_generate_for_all_patterns(self, tmp_path: Path):
        """Test code generation for each pattern subtype."""
        generator = CppGenerator(GenerationConfig(
            output_dir=tmp_path,
            library_name="test_integration",
        ))

        test_cases = [
            (
                HotspotMethod(name="com.example.VectorOps.map", signature="([D)V"),
                "vector_expr", "vector_map",
            ),
            (
                HotspotMethod(name="com.example.MathOps.sigmoid", signature="(D)D"),
                "math", "math_sigmoid",
            ),
            (
                HotspotMethod(name="com.example.StringOps.split", signature="(Ljava/lang/String;)[Ljava/lang/String;"),
                "string", "string_split",
            ),
            (
                HotspotMethod(name="com.example.BufferOps.copy", signature="([B[B)V"),
                "memory_bandwidth", "array_copy",
            ),
        ]

        for method, pattern_type, pattern_subtype in test_cases:
            classification = PatternClassification(
                method=method,
                pattern_type=pattern_type,
                confidence=0.8,
                matched_rules=["test"],
                pattern_subtype=pattern_subtype,
            )

            output_file = generator.generate_for_method(method, classification)
            assert output_file.exists()
            assert output_file.suffix == ".cpp"
            content = output_file.read_text()
            assert "JNIEXPORT" in content
            assert "Java_" in content

    def test_generate_bridge_and_cmake(self, tmp_path: Path):
        """Test JNI bridge and CMake generation."""
        generator = CppGenerator(GenerationConfig(
            output_dir=tmp_path,
            library_name="test_bridge",
        ))

        methods = [
            (
                HotspotMethod(name="com.example.Ops.method1", signature="(D)D"),
                PatternClassification(
                    method=HotspotMethod(name="com.example.Ops.method1", signature="(D)D"),
                    pattern_type="math",
                    confidence=0.9,
                    matched_rules=[],
                    pattern_subtype="generic",
                ),
            ),
        ]

        for method, classification in methods:
            generator.generate_for_method(method, classification)

        bridge = generator.generate_jni_bridge(methods)
        assert bridge.exists()
        bridge_content = bridge.read_text()
        assert "Java_" in bridge_content

        cmake = generator.generate_cmake([bridge])
        assert cmake.exists()
        cmake_content = cmake.read_text()
        assert "add_library" in cmake_content
        assert "test_bridge" in cmake_content


class TestCompilerIntegration:
    """Integration tests for compilation."""

    def test_compile_single_file(self, tmp_path: Path):
        """Test compiling a single C++ file."""
        cpp_file = tmp_path / "test.cpp"
        cpp_file.write_text("""
#include <jni.h>
extern "C"
JNIEXPORT jint JNICALL
Java_test_Class_method(JNIEnv* env, jobject thiz) {
    return 42;
}
""")

        compiler = Compiler(build_dir=tmp_path / "build")
        result = compiler.compile_single(cpp_file)
        assert result is not None


class TestAcceleratorIntegration:
    """End-to-end integration tests for HotspotAccelerator."""

    def test_accelerator_run_mock(self, tmp_path: Path, sample_profile):
        """Test accelerator run with mocked JFR."""
        with mock.patch(
            "arkprobe.hotspot.accelerator.JfrHotspotProfiler.profile",
            return_value=sample_profile
        ):
            config = AccelerationConfig(
                output_dir=tmp_path / "output",
                library_name="test_accel",
            )
            accelerator = HotspotAccelerator(jvm_pid=99999, config=config)

            with mock.patch.object(
                accelerator.bytecode_extractor,
                "get_method_bytecode",
                return_value="b2 00 01 00 00"
            ):
                result = accelerator.run(profiling_duration=1, run_benchmark=False)

        assert result.methods_analyzed == 3
        assert len(result.generated_files) > 0

        output_json = config.output_dir / "acceleration_result.json"
        assert output_json.exists()

    def test_accelerator_generate_report(self, tmp_path: Path):
        """Test report generation."""
        result = AccelerationResult()
        result.methods_analyzed = 10
        result.methods_classified = 5
        result.methods_accelerated = 3
        result.recommended_methods = [
            {
                "method": "com.example.FastMath.compute",
                "speedup": 2.5,
                "java_time_ms": 10.0,
                "cpp_time_ms": 4.0,
            },
        ]

        config = AccelerationConfig(output_dir=tmp_path)
        accelerator = HotspotAccelerator(jvm_pid=12345, config=config)

        report = accelerator.generate_report(result)
        assert "Hotspot Acceleration Report" in report
        assert "Methods analyzed: 10" in report
        assert "com.example.FastMath.compute" in report


class TestBenchmarkRunnerIntegration:
    """Integration tests for benchmarking."""

    @mock.patch("arkprobe.hotspot.runtime.jni_loader.run_cmd")
    @mock.patch("arkprobe.hotspot.runtime.jni_loader.JNILoader.load_library")
    def test_benchmark_runner_placeholder(self, mock_load, mock_run_cmd):
        """Test benchmark runner with mocked method."""
        mock_load.return_value = True

        runner = BenchmarkRunner(jvm_pid=12345)

        method = HotspotMethod(
            name="test.Example.compute",
            signature="(D)D",
        )

        with mock.patch("pathlib.Path.exists", return_value=True):
            result = runner.benchmark_method(
                method,
                cpp_so_path=Path("/nonexistent/lib.so"),
                iterations=10,
                warmup_iters=5,
            )

        assert result.method_name == "test.Example.compute"
        assert result.iterations == 10
        assert result.speedup_factor > 0


class TestFullPipeline:
    """Full pipeline integration tests."""

    def test_end_to_end_with_mocks(self, tmp_path: Path):
        """Complete end-to-end test with all components mocked."""
        methods = [
            HotspotMethod(
                name="com.example.test.Add",
                signature="(II)I",
                cpu_time_ns=20_000_000,
                cpu_time_percent=6.0,
            ),
        ]

        profile = HotspotProfile(
            pid=12345,
            jdk_version="openjdk 17",
            duration_sec=30,
            methods=methods,
        )

        config = AccelerationConfig(
            output_dir=tmp_path / "accel_output",
            min_cpu_percent=2.0,
            min_speedup=1.5,
        )

        accelerator = HotspotAccelerator(jvm_pid=12345, config=config)

        with mock.patch.object(
            accelerator.profiler,
            "profile",
            return_value=profile
        ):
            with mock.patch.object(
                accelerator.bytecode_extractor,
                "get_method_bytecode",
                return_value="deadbeef"
            ):
                mock_compile_result = mock.MagicMock()
                mock_compile_result.success = True
                mock_compile_result.output_file = config.output_dir / "libtest.so"

                with mock.patch.object(
                    accelerator.compiler,
                    "compile",
                    return_value=mock_compile_result
                ):
                    result = accelerator.run(profiling_duration=1, run_benchmark=False)

        assert result.methods_analyzed == 1
        assert result.generated_files

    def test_convenience_function(self, tmp_path: Path, sample_profile):
        """Test the accelerate() convenience function."""
        with mock.patch(
            "arkprobe.hotspot.accelerator.JfrHotspotProfiler.profile",
            return_value=sample_profile
        ):
            with mock.patch(
                "arkprobe.hotspot.accelerator.BytecodeExtractor.get_method_bytecode",
                return_value="deadbeef"
            ):
                with mock.patch(
                    "arkprobe.hotspot.accelerator.Compiler.compile"
                ) as mock_compile:
                    mock_compile.return_value = mock.MagicMock(success=False, output_file=None)
                    result = accelerate(
                        jvm_pid=12345,
                        output_dir=tmp_path / "conv_test",
                        profiling_duration=1,
                        run_benchmark=False,
                    )

        assert isinstance(result, AccelerationResult)


class TestMemoryBandwidthPattern:
    """Tests for memory bandwidth pattern classification."""

    def test_memory_bandwidth_methods_classify_correctly(self):
        """Test that memory bandwidth patterns are recognized."""
        matcher = PatternMatcher()

        test_cases = [
            ("java.util.Arrays.arraycopy", "(Ljava/lang/Object;ILjava/lang/Object;II)V", "memory_bandwidth"),
            ("java.nio.ByteBuffer.put", "([B)V", "memory_bandwidth"),
            ("com.example.BufferOps.scale", "([F)V", "memory_bandwidth"),
        ]

        for method_name, signature, expected_type in test_cases:
            method = HotspotMethod(
                name=method_name,
                signature=signature,
                cpu_time_percent=3.0,
            )
            classification = matcher.classify(method)
            assert classification.pattern_type == expected_type


class TestPatternSubtypes:
    """Tests for pattern subtype inference."""

    def test_memory_bandwidth_subtypes(self):
        """Test memory bandwidth subtype detection."""
        matcher = PatternMatcher()

        # Test array_copy directly
        method1 = HotspotMethod(
            name="java.util.Arrays.arraycopy",
            signature="(Ljava/lang/Object;ILjava/lang/Object;II)V",
            cpu_time_percent=3.0,
        )
        class1 = matcher.classify(method1)
        assert class1.pattern_subtype == "array_copy"

        # Test array_scale: needs 2 arrays in signature + scale/mul in name
        method2 = HotspotMethod(
            name="com.example.Ops.scale",
            signature="([D[D)V",
            cpu_time_percent=3.0,
        )
        class2 = matcher.classify(method2)
        assert class2.pattern_subtype == "array_scale"

        # Test array_add
        method3 = HotspotMethod(
            name="com.example.Ops.add",
            signature="([F[F)V",
            cpu_time_percent=3.0,
        )
        class3 = matcher.classify(method3)
        assert class3.pattern_subtype == "array_add"

    def test_math_subtypes(self):
        """Test math pattern subtype detection."""
        matcher = PatternMatcher()

        test_cases = [
            ("com.example.Math.sigmoid", "math_sigmoid"),
            ("com.example.Math.relu", "math_relu"),
            ("com.example.Math.gemm", "math_gemm"),
        ]

        for method_name, expected_subtype in test_cases:
            method = HotspotMethod(
                name=method_name,
                signature="(D)D",
                cpu_time_percent=2.0,
            )
            classification = matcher.classify(method)
            assert classification.pattern_subtype == expected_subtype

    def test_string_subtypes(self):
        """Test string pattern subtype detection."""
        matcher = PatternMatcher()

        test_cases = [
            ("com.example.String.split", "string_split"),
            ("com.example.String.replace", "string_replace"),
            ("com.example.String.parseInt", "string_parse"),
        ]

        for method_name, expected_subtype in test_cases:
            method = HotspotMethod(
                name=method_name,
                signature="(Ljava/lang/String;)I",
                cpu_time_percent=3.0,
            )
            classification = matcher.classify(method)
            assert classification.pattern_subtype == expected_subtype
