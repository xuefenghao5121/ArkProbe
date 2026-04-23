"""Tests for the hotspot code generator."""

from __future__ import annotations

import pytest
from pathlib import Path
from arkprobe.hotspot import (
    HotspotMethod,
    PatternClassification,
    CppGenerator,
    GenerationConfig,
    generate_cpp_code,
    PatternMatcher,
    classify_hotspot_method,
)


class TestCppGenerator:
    """Test C++ code generation for hotspot methods."""

    def test_generate_vector_expr_code(self, tmp_path: Path):
        method = HotspotMethod(
            name="com.example.FastMath.compute",
            signature="(II)D",
            cpu_time_ns=15_200_000,
        )
        classification = PatternClassification(
            method=method,
            pattern_type="vector_expr",
            confidence=0.85,
            matched_rules=["method_name:compute", "signature:\\(II\\)D"],
        )

        config = GenerationConfig(output_dir=tmp_path, library_name="test_hotspot")
        generator = CppGenerator(config)

        output_file = generator.generate_for_method(method, classification)
        assert output_file.exists()
        content = output_file.read_text()
        assert "NEON" in content or "arm_neon" in content or "vector" in content.lower()

    def test_generate_math_code(self, tmp_path: Path):
        method = HotspotMethod(
            name="com.example.NumericUtils.sigmoid",
            signature="(D)D",
            cpu_time_ns=8_000_000,
        )
        classification = PatternClassification(
            method=method,
            pattern_type="math",
            confidence=0.9,
            matched_rules=["method_name:sigmoid"],
        )

        config = GenerationConfig(output_dir=tmp_path)
        generator = CppGenerator(config)
        output_file = generator.generate_for_method(method, classification)

        content = output_file.read_text()
        assert "sigmoid" in content or "exp" in content

    def test_generate_unknown_code(self, tmp_path: Path):
        method = HotspotMethod(
            name="java.lang.Object.hashCode",
            signature="()I",
            cpu_time_ns=1_000_000,
        )
        classification = PatternClassification(
            method=method,
            pattern_type="unknown",
            confidence=0.0,
            matched_rules=[],
        )

        config = GenerationConfig(output_dir=tmp_path)
        generator = CppGenerator(config)
        output_file = generator.generate_for_method(method, classification)

        content = output_file.read_text()
        assert "TODO" in content or "stub" in content.lower()

    def test_generate_jni_bridge(self, tmp_path: Path):
        methods = []
        for i, (name, sig, ptype) in enumerate([
            ("com.example.FastMath.compute", "(II)D", "vector_expr"),
            ("com.example.NumericUtils.sigmoid", "(D)D", "math"),
        ]):
            m = HotspotMethod(name=name, signature=sig, cpu_time_ns=1_000_000 * (i + 1))
            c = PatternClassification(m, ptype, 0.8, [])
            methods.append((m, c))

        config = GenerationConfig(output_dir=tmp_path, library_name="test_bridge")
        generator = CppGenerator(config)
        bridge_file = generator.generate_jni_bridge(methods)

        assert bridge_file.exists()
        content = bridge_file.read_text()
        assert "JNI_OnLoad" in content
        assert "RegisterNatives" in content

    def test_generate_cmake(self, tmp_path: Path):
        config = GenerationConfig(output_dir=tmp_path, library_name="test_lib")
        generator = CppGenerator(config)
        cmake_file = generator.generate_cmake(["method1.cpp", "method2.cpp"])

        assert cmake_file.exists()
        content = cmake_file.read_text()
        assert "cmake_minimum_required" in content
        assert "test_lib" in content
        assert "find_package(JNI" in content

    def test_mangle_class_name(self, tmp_path: Path):
        config = GenerationConfig(output_dir=tmp_path)
        generator = CppGenerator(config)

        assert generator._mangle_class_name("com.example.FastMath") == "com_example_FastMath"
        assert generator._mangle_class_name("java.lang.String") == "java_lang_String"
        assert generator._mangle_class_name("a.b.c$Inner") == "a_b_c_00024Inner"

    def test_infer_params(self, tmp_path: Path):
        config = GenerationConfig(output_dir=tmp_path)
        generator = CppGenerator(config)

        # Test primitive parameter parsing
        params = generator._infer_params("(IDJ)V")
        assert len(params) == 3
        assert params[0]["jni_type"] == "jint"
        assert params[1]["jni_type"] == "jdouble"
        assert params[2]["jni_type"] == "jlong"

    def test_infer_return_type(self, tmp_path: Path):
        config = GenerationConfig(output_dir=tmp_path)
        generator = CppGenerator(config)

        assert generator._infer_return_type("()V") == "void"
        assert generator._infer_return_type("()I") == "jint"
        assert generator._infer_return_type("()D") == "jdouble"
        assert generator._infer_return_type("()Ljava/lang/String;") == "jstring"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
