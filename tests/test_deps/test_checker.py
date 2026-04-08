"""Tests for the dependency checker module."""

from unittest.mock import patch

import pytest

from arkprobe.deps.checker import (
    check_binary,
    check_dependencies,
    check_all_available,
    format_missing_deps,
)
from arkprobe.deps.registry import get_install_hint, INSTALL_HINTS


class TestCheckBinary:
    def test_available_binary(self):
        # python should always be available
        result = check_binary("python3")
        assert result.available is True
        assert result.path != ""
        assert result.binary == "python3"

    def test_missing_binary(self):
        result = check_binary("nonexistent_binary_xyz_12345")
        assert result.available is False
        assert result.path == ""
        assert result.install_hint != ""

    @patch("arkprobe.deps.checker.shutil.which")
    def test_mock_available(self, mock_which):
        mock_which.return_value = "/usr/bin/sysbench"
        result = check_binary("sysbench")
        assert result.available is True
        assert result.path == "/usr/bin/sysbench"

    @patch("arkprobe.deps.checker.shutil.which")
    def test_mock_missing(self, mock_which):
        mock_which.return_value = None
        result = check_binary("sysbench")
        assert result.available is False


class TestCheckDependencies:
    @patch("arkprobe.deps.checker.shutil.which")
    def test_all_available(self, mock_which):
        mock_which.return_value = "/usr/bin/fake"
        results = check_dependencies(["sysbench", "mysqld"])
        assert len(results) == 2
        assert all(r.available for r in results)

    @patch("arkprobe.deps.checker.shutil.which")
    def test_some_missing(self, mock_which):
        mock_which.side_effect = lambda name: "/usr/bin/mysqld" if name == "mysqld" else None
        results = check_dependencies(["sysbench", "mysqld"])
        assert results[0].available is False
        assert results[1].available is True

    def test_empty_list(self):
        results = check_dependencies([])
        assert results == []


class TestCheckAllAvailable:
    @patch("arkprobe.deps.checker.shutil.which")
    def test_all_present(self, mock_which):
        mock_which.return_value = "/usr/bin/fake"
        assert check_all_available(["a", "b"]) is True

    @patch("arkprobe.deps.checker.shutil.which")
    def test_one_missing(self, mock_which):
        mock_which.side_effect = lambda name: "/usr/bin/a" if name == "a" else None
        assert check_all_available(["a", "b"]) is False

    def test_empty_means_available(self):
        assert check_all_available([]) is True


class TestFormatMissingDeps:
    @patch("arkprobe.deps.checker.shutil.which")
    def test_format_with_missing(self, mock_which):
        mock_which.return_value = None
        results = check_dependencies(["sysbench", "mysqld"])
        msg = format_missing_deps(results)
        assert "sysbench" in msg
        assert "mysqld" in msg
        assert "Missing dependencies:" in msg

    @patch("arkprobe.deps.checker.shutil.which")
    def test_format_all_available(self, mock_which):
        mock_which.return_value = "/usr/bin/fake"
        results = check_dependencies(["sysbench"])
        msg = format_missing_deps(results)
        assert msg == ""


class TestRegistry:
    def test_known_binary(self):
        hint = get_install_hint("sysbench")
        assert "yum" in hint or "install" in hint.lower()

    def test_unknown_binary(self):
        hint = get_install_hint("totally_unknown_tool")
        assert "totally_unknown_tool" in hint

    def test_registry_has_common_tools(self):
        expected = ["sysbench", "mysqld", "redis-server", "ffmpeg", "nginx", "perf"]
        for tool in expected:
            assert tool in INSTALL_HINTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
