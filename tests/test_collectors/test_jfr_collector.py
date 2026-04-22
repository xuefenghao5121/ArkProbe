"""Tests for JFR collector and JDK version detection."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from arkprobe.collectors.base import CollectionResult
from arkprobe.collectors.jfr_collector import (
    JfrCollector,
    DEFAULT_JFR_PROFILES,
    JFR_EVENT_PROFILES,
    detect_jdk_version,
    resolve_jfr_events,
    _major_jdk_version,
)


@pytest.fixture
def collector(tmp_path: Path) -> JfrCollector:
    return JfrCollector(output_dir=tmp_path / "jfr")


class TestResolveJfrEvents:
    def test_default_profiles(self):
        events = resolve_jfr_events(DEFAULT_JFR_PROFILES)
        assert "jdk.GCPhaseLevel" in events
        assert "jdk.Compilation" in events
        assert "jdk.SafepointBegin" in events

    def test_single_profile(self):
        events = resolve_jfr_events(["gc"])
        assert "jdk.GCHeapSummary" in events
        assert "jdk.Compilation" not in events

    def test_custom_event(self):
        events = resolve_jfr_events(["jdk.ThreadSleep"])
        assert "jdk.ThreadSleep" in events


class TestMajorJdkVersion:
    def test_jdk8(self):
        assert _major_jdk_version("1.8.0_362") == 8

    def test_jdk11(self):
        assert _major_jdk_version("11.0.18") == 11

    def test_jdk17(self):
        assert _major_jdk_version("17.0.6") == 17

    def test_jdk21(self):
        assert _major_jdk_version("21.0.1") == 21

    def test_unknown(self):
        assert _major_jdk_version("unknown") == 0


class TestJfrCollector:
    def test_collect_no_pid(self, collector: JfrCollector):
        result = collector.collect()
        assert result.collector_name == "jfr"
        assert len(result.errors) == 1
        assert "No target_pid" in result.errors[0]

    @patch("arkprobe.collectors.jfr_collector.run_cmd")
    @patch("arkprobe.collectors.jfr_collector.detect_jdk_version")
    def test_collect_jdk8_fallback(self, mock_detect, mock_run, collector: JfrCollector):
        mock_detect.return_value = "1.8.0_362"

        jstat_result = MagicMock()
        jstat_result.ok = True
        jstat_result.stdout = "  EC       EU       OC       OU       MC       MU\n102400.0 51200.0 204800.0 102400.0 10240.0 8192.0\n102400.0 51200.0 204800.0 153600.0 10240.0 8192.0"

        jstack_result = MagicMock()
        jstack_result.ok = True
        jstack_result.stdout = '"main" #1 prio=5 os_prio=0 tid=0x... daemon\n"GC Thread" #2 daemon\n"Thread-3" #4 java.lang.Thread.State: BLOCKED'

        gcutil_result = MagicMock()
        gcutil_result.ok = True
        gcutil_result.stdout = "  S0     S1     E      O      M     YGC   YGCT   FGC FGCT\n  0.00  50.00  30.00  75.00  80.00    10   0.500   1  0.200"

        def side_effect(cmd, **kwargs):
            if "jstat" in cmd and "-gcutil" in cmd:
                return gcutil_result
            if "jstat" in cmd:
                return jstat_result
            if "jstack" in cmd:
                return jstack_result
            return MagicMock(ok=False, stderr="not found")

        mock_run.side_effect = side_effect

        result = collector.collect(target_pid=12345, jfr_duration_sec=5)
        assert result.collector_name == "jfr"
        assert result.data.get("jfr_available") is False
        assert "jstat_parsed" in result.data
        assert "jstack_parsed" in result.data

    def test_parse_jstat_gc(self, collector: JfrCollector):
        output = "  EC       EU       OC       OU       MC       MU       YGC   YGCT    FGC FGCT\n102400.0 51200.0 204800.0 102400.0 10240.0 8192.0    10   0.500   1  0.200\n102400.0 51200.0 204800.0 153600.0 10240.0 8192.0    15   0.750   1  0.200"
        parsed = collector._parse_jstat_gc(output)
        assert parsed["YGC"] == 15.0
        assert parsed["FGC"] == 1.0
        assert parsed["_sample_count"] == 2

    def test_parse_jstat_gc_empty(self, collector: JfrCollector):
        parsed = collector._parse_jstat_gc("")
        assert parsed == {}

    def test_parse_jstack(self, collector: JfrCollector):
        output = '"main" #1 daemon\n"Thread-1" #2\n"GC Thread" #3 daemon\n"Thread-4" #4 java.lang.Thread.State: BLOCKED\nFound one Java-level deadlock'
        parsed = collector._parse_jstack(output)
        assert parsed["total_threads"] > 0
        assert parsed["daemon_threads"] >= 2
        assert parsed["blocked_threads"] >= 1
        assert parsed["deadlocked_threads"] == -1

    def test_parse_jfr_json(self, collector: JfrCollector):
        import json
        jfr_data = {
            "events": [
                {"type": "jdk.GCHeapSummary", "values": {"heapUsed": 1073741824}},
                {"type": "jdk.Compilation", "values": {"compiler": "c2"}},
                {"type": "jdk.SafepointBegin", "values": {}},
                {"type": "jdk.ThreadStart", "values": {"daemon": True}},
            ]
        }
        parsed = collector._parse_jfr_json(json.dumps(jfr_data))
        assert len(parsed["gc_events"]) == 1
        assert len(parsed["jit_events"]) == 1
        assert len(parsed["safepoint_events"]) == 1
        assert len(parsed["thread_events"]) == 1

    def test_parse_jfr_json_invalid(self, collector: JfrCollector):
        parsed = collector._parse_jfr_json("not valid json")
        assert parsed["gc_events"] == []
