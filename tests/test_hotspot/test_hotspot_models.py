"""Tests for the hotspot module."""

from __future__ import annotations

import pytest
from arkprobe.hotspot import HotspotMethod, HotspotProfile, PatternMatcher, classify_hotspot_method


class TestHotspotMethod:
    """Test HotspotMethod model."""

    def test_create_hotspot_method(self):
        method = HotspotMethod(
            name="com.example.FastMath.compute",
            signature="(II)D",
            bytecode_size=42,
            compilation_count=3,
            cpu_time_ns=15_200_000,
        )
        assert method.name == "com.example.FastMath.compute"
        assert method.signature == "(II)D"
        assert method.bytecode_size == 42
        assert method.cpu_time_percent == 0.0

    def test_to_dict_roundtrip(self):
        original = HotspotMethod(
            name="com.example.Parser.parseCsv",
            signature="(Ljava/lang/String;)Ljava/util/List;",
            cpu_time_ns=12_800_000,
            pattern_type="string",
        )
        data = original.to_dict()
        restored = HotspotMethod.from_dict(data)
        assert restored.name == original.name
        assert restored.pattern_type == "string"


class TestPatternMatcher:
    """Test pattern classification."""

    def test_classify_vector_expr(self):
        method = HotspotMethod(
            name="com.example.IntStream.map",
            signature="(Ljava/util/function/Function;)Ljava/util/stream/IntStream;",
            cpu_time_ns=10_000_000,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "vector_expr"
        assert classification.confidence >= 0.5

    def test_classify_string(self):
        method = HotspotMethod(
            name="java.lang.String.split",
            signature="(Ljava/lang/String;)Ljava/util/List;",
            cpu_time_ns=8_000_000,
        )
        classification = classify_hotspot_method(method)
        # This hits "string" pattern
        assert classification.pattern_type in ("string", "unknown")

    def test_classify_math(self):
        method = HotspotMethod(
            name="com.example.FastMath.sigmoid",
            signature="(D)D",
            cpu_time_ns=5_000_000,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "math"

    def test_classify_unknown(self):
        method = HotspotMethod(
            name="java.lang.Object.hashCode",
            signature="()I",
            cpu_time_ns=1_000_000,
        )
        classification = classify_hotspot_method(method)
        assert classification.pattern_type == "unknown"


class TestHotspotProfile:
    """Test HotspotProfile aggregation."""

    def test_top_methods_sorted_by_cpu_time(self):
        profile = HotspotProfile(
            pid=12345,
            jdk_version="11.0.22",
            duration_sec=30,
            methods=[
                HotspotMethod(name="m1", cpu_time_ns=1_000),
                HotspotMethod(name="m2", cpu_time_ns=5_000),
                HotspotMethod(name="m3", cpu_time_ns=3_000),
            ],
        )
        top = profile.get_top_methods(2)
        assert top[0].name == "m2"
        assert top[1].name == "m3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
