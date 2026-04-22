"""Loader fuzzy matching and short name resolution tests."""

import pytest
from pathlib import Path
from arkprobe.scenarios.loader import get_scenario_by_name, load_builtin_scenarios


class TestFuzzyMatching:
    """Test get_scenario_by_name fuzzy matching behavior."""

    def test_exact_match(self):
        sc = get_scenario_by_name("Compute Intensive")
        assert sc is not None
        assert "Compute Intensive" in sc.name

    def test_case_insensitive(self):
        sc = get_scenario_by_name("compute intensive")
        assert sc is not None
        assert "Compute Intensive" in sc.name

    def test_short_name_compute(self):
        sc = get_scenario_by_name("compute")
        assert sc is not None
        assert "Compute Intensive" in sc.name

    def test_short_name_memory(self):
        sc = get_scenario_by_name("memory")
        assert sc is not None
        assert "Memory Intensive" in sc.name

    def test_short_name_mixed(self):
        sc = get_scenario_by_name("mixed")
        assert sc is not None
        assert "Mixed Workload" in sc.name

    def test_short_name_stream(self):
        sc = get_scenario_by_name("stream")
        assert sc is not None
        assert "STREAM" in sc.name

    def test_short_name_random(self):
        sc = get_scenario_by_name("random")
        assert sc is not None
        assert "Random Access" in sc.name

    def test_short_name_crypto(self):
        sc = get_scenario_by_name("crypto")
        assert sc is not None
        assert "Cryptographic" in sc.name

    def test_short_name_compress(self):
        sc = get_scenario_by_name("compress")
        assert sc is not None
        assert "Compression" in sc.name

    def test_short_name_video(self):
        sc = get_scenario_by_name("video")
        assert sc is not None
        assert "Video" in sc.name

    def test_short_name_ml(self):
        sc = get_scenario_by_name("ml")
        assert sc is not None
        assert "ML Inference" in sc.name

    def test_short_name_oltp(self):
        sc = get_scenario_by_name("oltp")
        assert sc is not None
        assert "OLTP" in sc.name or "Database" in sc.name

    def test_short_name_kv(self):
        sc = get_scenario_by_name("kv")
        assert sc is not None
        assert "KV" in sc.name or "Key-Value" in sc.name

    def test_short_name_web(self):
        sc = get_scenario_by_name("web")
        assert sc is not None
        assert "Web" in sc.name

    def test_no_match_returns_none(self):
        sc = get_scenario_by_name("nonexistent_scenario_xyz")
        assert sc is None

    def test_underscore_hyphen_normalization(self):
        # "compute_intensive" -> compute -> should match
        sc = get_scenario_by_name("compute_intensive")
        assert sc is not None
        assert "Compute Intensive" in sc.name

        # "memory-intensive" -> memory -> should match
        sc = get_scenario_by_name("memory-intensive")
        assert sc is not None
        assert "Memory Intensive" in sc.name

    def test_substring_reverse_match(self):
        # Full name contains short name, bidirectional matching
        sc = get_scenario_by_name("compute")
        assert sc is not None


class TestBuiltinAllPresent:
    """Ensure all 13 builtin scenarios have working short names."""

    def test_all_builtin_scenarios_loadable(self):
        builtins = load_builtin_scenarios()
        assert len(builtins) == 13

    @pytest.mark.parametrize("short_name,expected_fragment", [
        ("compute", "Compute"),
        ("memory", "Memory"),
        ("mixed", "Mixed"),
        ("stream", "STREAM"),
        ("random", "Random"),
        ("crypto", "Cryptographic"),
        ("compress", "Compression"),
        ("video", "Video"),
        ("ml", "ML"),
        ("oltp", "Database"),
        ("kv", "Key-Value"),
        ("web", "Web"),
        ("jvm", "JVM"),
    ])
    def test_short_name_maps_to_builtin(self, short_name, expected_fragment):
        sc = get_scenario_by_name(short_name)
        assert sc is not None, f"Short name '{short_name}' should resolve to a scenario"
        assert sc.builtin is True, f"'{short_name}' should resolve to a builtin scenario"
        assert expected_fragment in sc.name, \
            f"Short name '{short_name}' resolved to '{sc.name}', expected to contain '{expected_fragment}'"
