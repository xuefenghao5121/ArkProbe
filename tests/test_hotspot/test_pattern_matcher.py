"""Tests for the pattern matcher."""

from __future__ import annotations

import pytest
from arkprobe.hotspot import HotspotMethod, PatternMatcher, classify_hotspot_method


class TestPatternMatcher:
    """Test hotspot method pattern classification."""

    def test_vector_expression_stream_map(self):
        method = HotspotMethod(
            name="com.example.DataProcessor.stream.map",
            signature="(Ljava/util/function/Function;)Ljava/util/stream/Stream;",
            cpu_time_percent=15.0,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "vector_expr"
        assert classification.confidence >= 0.7

    def test_vector_expression_array_ops(self):
        method = HotspotMethod(
            name="com.example.ArrayOps.process",
            signature="([D)D",
            cpu_time_percent=12.0,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "vector_expr"

    def test_string_split(self):
        method = HotspotMethod(
            name="com.example.CsvParser.parseLine",
            signature="(Ljava/lang/String;)V",
            cpu_time_percent=10.0,
        )
        classification = classify_hotspot_method(method)
        # "Parser" class + string operations → string pattern
        assert classification.pattern_type in ("string", "math", "unknown")

    def test_math_sigmoid(self):
        method = HotspotMethod(
            name="com.example.NeuralNet.sigmoid",
            signature="(D)D",
            cpu_time_percent=8.0,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "math"

    def test_math_sin_cos(self):
        method = HotspotMethod(
            name="com.example.TrigUtils.sin",
            signature="(D)D",
            cpu_time_percent=5.0,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "math"

    def test_jdk_internal_methods_filtered(self):
        """Test that JVM internal methods are classified as unknown."""
        method = HotspotMethod(
            name="java.lang.Thread.run",
            signature="()V",
            cpu_time_percent=1.0,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "unknown"

    def test_estimate_simd_potential(self):
        matcher = PatternMatcher()
        method = HotspotMethod(
            name="com.example.ArrayOps.map",
            signature="([D[D[D)V",
            cpu_time_percent=20.0,
        )
        potential = matcher.estimate_simd_potential(method)
        assert 0.0 <= potential <= 1.0
        # Array operations should have decent SIMD potential
        assert potential >= 0.5

    def test_estimate_deopt_risk(self):
        matcher = PatternMatcher()
        method = HotspotMethod(
            name="com.example.HotMethod.run",
            signature="()V",
            compilation_count=8,
            inline_count=150,
            bytecode_size=300,
        )
        risk = matcher.estimate_deopt_risk(method)
        # High compilation count + large bytecode = higher risk
        assert risk >= 0.5

    def test_deopt_risk_low_for_stable_methods(self):
        matcher = PatternMatcher()
        method = HotspotMethod(
            name="com.example.StableMethod.run",
            signature="()V",
            compilation_count=1,
            inline_count=5,
            bytecode_size=50,
        )
        risk = matcher.estimate_deopt_risk(method)
        assert risk <= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
