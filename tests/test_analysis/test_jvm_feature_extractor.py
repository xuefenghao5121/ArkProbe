"""Tests for JVM feature extraction from JFR/jstat data."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arkprobe.analysis.feature_extractor import FeatureExtractor
from arkprobe.collectors.base import CollectionResult
from arkprobe.model.enums import ScenarioType
from arkprobe.model.schema import JvmCharacteristics, GCMetrics, JITMetrics, JVMThreadMetrics


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor(kunpeng_model="920")


def _make_jfr_collection_result(**overrides) -> CollectionResult:
    defaults = {
        "collector_name": "jfr",
        "data": {
            "jdk_version": "17.0.6",
            "jfr_available": True,
            "jfr_events_collected": ["gc", "jit", "thread"],
            "jfr_parsed": {
                "gc_events": [
                    {"type": "jdk.GCConfiguration", "values": {"collectorName": "G1"}},
                    {"type": "jdk.GCHeapSummary", "values": {
                        "heapUsed": 4294967296,  # 4GB
                        "heapSpace": {"size": 8589934592},  # 8GB
                    }},
                    {"type": "jdk.OldGCPhase", "values": {"duration": 50000000}},  # 50ms
                    {"type": "jdk.YoungGCPhase", "values": {"duration": 10000000}},  # 10ms
                ],
                "jit_events": [
                    {"type": "jdk.Compilation", "values": {"compiler": "c2"}},
                    {"type": "jdk.Compilation", "values": {"compiler": "c1"}},
                    {"type": "jdk.Deoptimization", "values": {}},
                ],
                "safepoint_events": [
                    {"type": "jdk.SafepointBegin", "values": {"duration": 5000000}},
                ],
                "thread_events": [
                    {"type": "jdk.ThreadStart", "values": {"daemon": True}},
                    {"type": "jdk.ThreadStart", "values": {"daemon": False}},
                ],
            },
        },
        "raw_files": {},
        "errors": [],
    }
    defaults["data"].update(overrides)
    return CollectionResult(**defaults)


def _make_jstat_collection_result(**overrides) -> CollectionResult:
    defaults = {
        "collector_name": "jfr",
        "data": {
            "jdk_version": "1.8.0_362",
            "jfr_available": False,
            "jstat_parsed": {
                "YGC": 20.0,
                "YGCT": 1.5,
                "FGC": 2.0,
                "FGCT": 0.5,
                "OU": 204800.0,
                "OC": 409600.0,
                "EU": 51200.0,
                "EC": 102400.0,
                "MU": 8192.0,
                "_sample_count": 60,
            },
            "gcutil_parsed": {
                "YGC": 20.0,
                "FGC": 2.0,
                "O": 75.0,
            },
            "jstack_parsed": {
                "total_threads": 150,
                "daemon_threads": 30,
                "deadlocked_threads": 0,
            },
        },
        "raw_files": {},
        "errors": [],
    }
    defaults["data"].update(overrides)
    return CollectionResult(**defaults)


class TestExtractJvmFromJfr:
    def test_basic_extraction(self, extractor: FeatureExtractor):
        result = _make_jfr_collection_result()
        jvm = extractor._extract_jvm(result)

        assert isinstance(jvm, JvmCharacteristics)
        assert jvm.jdk_version == "17.0.6"
        assert jvm.jfr_available is True
        assert jvm.jfr_events_collected == ["gc", "jit", "thread"]

    def test_gc_extraction(self, extractor: FeatureExtractor):
        result = _make_jfr_collection_result()
        jvm = extractor._extract_jvm(result)

        assert isinstance(jvm.gc, GCMetrics)
        assert jvm.gc.gc_algorithm == "G1"
        assert jvm.gc.heap_max_mb > 0
        assert jvm.gc.heap_usage_ratio > 0

    def test_jit_extraction(self, extractor: FeatureExtractor):
        result = _make_jfr_collection_result()
        jvm = extractor._extract_jvm(result)

        assert isinstance(jvm.jit, JITMetrics)
        assert jvm.jit.total_compilations == 2
        assert jvm.jit.c1_count == 1
        assert jvm.jit.c2_count == 1
        assert jvm.jit.deoptimization_count == 1

    def test_thread_extraction(self, extractor: FeatureExtractor):
        result = _make_jfr_collection_result()
        jvm = extractor._extract_jvm(result)

        assert isinstance(jvm.threads, JVMThreadMetrics)
        assert jvm.threads.total_threads == 2
        assert jvm.threads.daemon_threads == 1


class TestExtractJvmFromJstat:
    def test_basic_extraction(self, extractor: FeatureExtractor):
        result = _make_jstat_collection_result()
        jvm = extractor._extract_jvm(result)

        assert isinstance(jvm, JvmCharacteristics)
        assert jvm.jfr_available is False

    def test_gc_extraction(self, extractor: FeatureExtractor):
        result = _make_jstat_collection_result()
        jvm = extractor._extract_jvm(result)

        assert jvm.gc.young_gc_count == 20
        assert jvm.gc.full_gc_count == 2
        assert jvm.gc.heap_usage_ratio > 0

    def test_thread_extraction(self, extractor: FeatureExtractor):
        result = _make_jstat_collection_result()
        jvm = extractor._extract_jvm(result)

        assert jvm.threads.total_threads == 150
        assert jvm.threads.daemon_threads == 30

    def test_jit_defaults(self, extractor: FeatureExtractor):
        result = _make_jstat_collection_result()
        jvm = extractor._extract_jvm(result)

        assert jvm.jit.total_compilations == 0  # No JIT data from jstat


class TestExtractJvmEdgeCases:
    def test_empty_jfr_parsed(self, extractor: FeatureExtractor):
        result = _make_jfr_collection_result(jfr_parsed={})
        jvm = extractor._extract_jvm(result)
        assert jvm.gc.gc_algorithm == "unknown"
        assert jvm.gc.young_gc_count == 0

    def test_missing_jfr_parsed(self, extractor: FeatureExtractor):
        result = _make_jfr_collection_result()
        result.data.pop("jfr_parsed", None)
        jvm = extractor._extract_jvm(result)
        assert jvm.gc.gc_algorithm == "unknown"
