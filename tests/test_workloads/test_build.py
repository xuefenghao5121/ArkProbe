"""Tests for the workload build system."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from arkprobe.workloads.build import (
    SRC_DIR,
    WORKLOAD_SOURCES,
    resolve_builtin_command,
    get_workload_binary,
)


class TestSourceFiles:
    def test_source_dir_exists(self):
        assert SRC_DIR.exists(), f"Source directory not found: {SRC_DIR}"

    def test_all_sources_exist(self):
        for name, src_file in WORKLOAD_SOURCES.items():
            src_path = SRC_DIR / src_file
            assert src_path.exists(), f"Source file missing: {src_path}"

    def test_sources_are_c_files(self):
        for src_file in WORKLOAD_SOURCES.values():
            assert src_file.endswith(".c")


class TestResolveBuiltinCommand:
    @patch("arkprobe.workloads.build.get_workload_binary")
    def test_resolve_with_binary(self, mock_get):
        mock_get.return_value = Path("/cache/arkprobe_compute")
        result = resolve_builtin_command(
            "{builtin_binary_compute} --threads 4 --duration 60"
        )
        assert result == "/cache/arkprobe_compute --threads 4 --duration 60"

    @patch("arkprobe.workloads.build.get_workload_binary")
    def test_resolve_fallback(self, mock_get):
        mock_get.return_value = None
        result = resolve_builtin_command(
            "{builtin_binary_compute} --threads 4 --duration 60"
        )
        assert "python -m arkprobe.workloads.fallback" in result
        assert "--workload compute" in result

    def test_no_placeholder(self):
        cmd = "sysbench --threads 4"
        assert resolve_builtin_command(cmd) == cmd

    @patch("arkprobe.workloads.build.get_workload_binary")
    def test_multiple_placeholders(self, mock_get):
        mock_get.side_effect = lambda name: Path(f"/cache/arkprobe_{name}")
        result = resolve_builtin_command(
            "{builtin_binary_compute} && {builtin_binary_memory}"
        )
        assert "/cache/arkprobe_compute" in result
        assert "/cache/arkprobe_memory" in result


class TestGetWorkloadBinary:
    @patch("arkprobe.workloads.build._get_binary_path")
    def test_cached_binary_returned(self, mock_path):
        mock_bin = MagicMock(spec=Path)
        mock_bin.exists.return_value = True
        mock_path.return_value = mock_bin
        result = get_workload_binary("compute")
        assert result == mock_bin

    def test_unknown_workload(self):
        result = get_workload_binary("nonexistent_workload_xyz")
        assert result is None


class TestScenarioLoader:
    """Test that builtin scenario YAMLs load correctly."""

    def test_builtin_scenarios_loadable(self):
        from arkprobe.scenarios.loader import load_builtin_scenarios
        scenarios = load_builtin_scenarios()
        assert len(scenarios) == 13
        names = {s.name for s in scenarios}
        assert "Compute Intensive (builtin)" in names
        assert "Memory Intensive (builtin)" in names
        assert "Mixed Workload (builtin)" in names
        assert "STREAM Benchmark (builtin)" in names
        assert "Random Access Pattern (builtin)" in names
        assert "Database OLTP Micro-Kernel (builtin)" in names
        assert "Key-Value Store Micro-Kernel (builtin)" in names
        assert "Web Server Micro-Kernel (builtin)" in names
        assert "Cryptographic Operations (builtin)" in names
        assert "Compression Operations (builtin)" in names
        assert "Video Encoding (builtin)" in names
        assert "ML Inference (builtin)" in names

    def test_builtin_scenarios_have_no_deps(self):
        from arkprobe.scenarios.loader import load_builtin_scenarios
        for s in load_builtin_scenarios():
            assert s.dependencies == [], f"{s.name} should have no dependencies"
            assert s.builtin is True

    def test_lightweight_listing_never_errors(self):
        from arkprobe.scenarios.loader import list_scenarios_lightweight
        # Should never raise, even if YAMLs are malformed
        items = list_scenarios_lightweight()
        assert isinstance(items, list)
        assert len(items) > 0

    def test_lightweight_listing_includes_builtin(self):
        from arkprobe.scenarios.loader import list_scenarios_lightweight
        items = list_scenarios_lightweight()
        builtin_items = [i for i in items if i["builtin"]]
        assert len(builtin_items) == 13

    def test_external_scenarios_have_deps(self):
        from arkprobe.scenarios.loader import list_scenarios_lightweight
        items = list_scenarios_lightweight()
        external = [i for i in items if not i["builtin"]]
        # All external scenarios should have dependencies declared
        for item in external:
            assert len(item["dependencies"]) > 0, \
                f"{item['name']} should have dependencies declared"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
