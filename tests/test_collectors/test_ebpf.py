"""Tests for the eBPF collector module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from arkprobe.collectors.ebpf_collector import EbpfCollector
from arkprobe.collectors.base import CollectionResult


class TestEbpfCollector:
    """Test cases for EbpfCollector."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory."""
        return tmp_path / "ebpf_test"

    @pytest.fixture
    def collector(self, temp_output_dir):
        """Create an EbpfCollector instance."""
        return EbpfCollector(output_dir=temp_output_dir)

    def test_init(self, collector, temp_output_dir):
        """Test collector initialization."""
        assert collector.output_dir == temp_output_dir
        assert collector.backend == "auto"
        assert collector.BCC_TOOL_BASES is not None
        assert len(collector.BCC_TOOL_BASES) == 7
        assert len(collector.BCC_TOOLS) == 7

    def test_bcc_tools_list(self):
        """Verify BCC tools list is correct."""
        expected_bases = [
            "biolatency",
            "cachestat",
            "tcprtt",
            "tcpconnlat",
            "offcputime",
            "runqlat",
            "runqlen",
        ]
        expected_canonical = [f"{b}-bpfcc" for b in expected_bases]
        assert EbpfCollector.BCC_TOOL_BASES == expected_bases
        assert EbpfCollector.BCC_TOOLS == expected_canonical

    def test_resolve_bcc_tool_bpfcc_suffix(self):
        """Test _resolve_bcc_tool prefers -bpfcc variant when available."""
        with patch("arkprobe.collectors.ebpf_collector.shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/biolatency-bpfcc" if "-bpfcc" in x else None
            path = EbpfCollector._resolve_bcc_tool("biolatency")
            assert path == "biolatency-bpfcc"

    def test_resolve_bcc_tool_openeuler_path(self):
        """Test _resolve_bcc_tool falls back to openEuler path when -bpfcc not found."""
        def fake_which(x):
            if x == "biolatency-bpfcc":
                return None
            if x == "/usr/share/bcc/tools/biolatency":
                return "/usr/share/bcc/tools/biolatency"
            return None
        with patch("arkprobe.collectors.ebpf_collector.shutil.which", side_effect=fake_which):
            path = EbpfCollector._resolve_bcc_tool("biolatency")
            assert path == "/usr/share/bcc/tools/biolatency"

    def test_get_tool_path_caches(self):
        """Test _get_tool_path caches resolved paths."""
        collector = EbpfCollector(output_dir=Path("/tmp"))
        with patch.object(collector, "_resolve_bcc_tool", return_value="/usr/share/bcc/tools/biolatency") as mock_resolve:
            path1 = collector._get_tool_path("biolatency")
            path2 = collector._get_tool_path("biolatency")
            assert path1 == "/usr/share/bcc/tools/biolatency"
            assert path2 == "/usr/share/bcc/tools/biolatency"
            mock_resolve.assert_called_once()  # second call should use cache

    @patch("arkprobe.collectors.ebpf_collector.shutil.which")
    def test_is_available_all_tools_present(self, mock_which, collector):
        """Test is_available returns True when all tools present."""
        collector._bcc_available = None  # Reset cache
        collector._bpftrace_available = None
        mock_which.return_value = "/usr/bin/tool"
        available, reason = collector.is_available()
        assert available is True
        assert reason == ""

    @patch("arkprobe.collectors.ebpf_collector.EbpfCollector._check_bcc")
    @patch("arkprobe.collectors.ebpf_collector.run_cmd")
    def test_is_available_bpftrace_fallback(self, mock_run_cmd, mock_check_bcc, collector):
        """Test is_available returns True with bpftrace fallback when no BCC tools."""
        collector._bcc_available = None
        collector._bpftrace_available = None
        # Force _check_bcc to return False so we fall through to bpftrace check
        mock_check_bcc.return_value = False

        def run_cmd_side_effect(cmd, *args, **kwargs):
            if cmd[0] == "which" and cmd[1] == "bpftrace":
                return MagicMock(ok=True, stdout="", stderr="")
            return MagicMock(ok=False, stdout="", stderr="")

        mock_run_cmd.side_effect = run_cmd_side_effect
        available, reason = collector.is_available()
        assert available is True
        assert reason == "bpftrace fallback"

    @patch("arkprobe.collectors.ebpf_collector.shutil.which")
    @patch("arkprobe.collectors.ebpf_collector.run_cmd")
    def test_is_available_nothing_available(self, mock_run_cmd, mock_which, collector):
        """Test is_available returns False when no tools present."""
        # Need to reset cached check
        collector._bcc_available = None
        collector._bpftrace_available = None
        mock_which.return_value = None
        mock_run_cmd.return_value = MagicMock(ok=False)
        available, reason = collector.is_available()
        assert available is False
        assert "Missing tools:" in reason
        assert "biolatency-bpfcc" in reason

    def test_collect_graceful_degradation(self, collector):
        """Test collect returns error when tools unavailable."""
        with patch.object(collector, "is_available") as mock_avail:
            mock_avail.return_value = (False, "Missing tools: biolatency-bpfcc")
            result = collector.collect(duration_sec=1)
            assert isinstance(result, CollectionResult)
            assert result.collector_name == "ebpf"
            assert len(result.errors) > 0
            assert "Missing tools" in result.errors[0]

    @patch("arkprobe.collectors.ebpf_collector.run_cmd")
    def test_trace_io_latency_success(self, mock_run_cmd, collector):
        """Test trace_io_latency parses output correctly."""
        mock_run_cmd.return_value = MagicMock(
            ok=True,
            stdout="""    1.0 us      1  |
    2.0 us      5  |
    4.0 us     15  |||
    8.0 us     50  ||||||||
""",
            stderr="",
        )
        result = collector.trace_io_latency(duration_sec=1)
        assert "histogram" in result
        assert "avg_latency_us" in result
        assert "p99_latency_us" in result

    @patch("arkprobe.collectors.ebpf_collector.run_cmd")
    def test_trace_cache_stats(self, mock_run_cmd, collector):
        """Test trace_cache_stats parses output correctly."""
        mock_run_cmd.return_value = MagicMock(
            ok=True,
            stdout="""    PERCENTAGE    HIT     MISS    DIRTIES   READ_HIT%   WRITE_HIT%
      0.00%    10000     500       100       95.24%       4.76%
      1.00%    12000     600       120       95.00%       5.00%
""",
            stderr="",
        )
        result = collector.trace_cache_stats(duration_sec=1)
        assert "page_cache_hits" in result
        assert "page_cache_misses" in result
        assert "hit_rate" in result

    @patch("arkprobe.collectors.ebpf_collector.run_cmd")
    def test_trace_tcp_latency_fallback(self, mock_run_cmd, collector):
        """Test tcp latency uses fallback when primary fails."""
        # First call fails (tcprtt), second succeeds (tcpconnlat)
        mock_run_cmd.side_effect = [
            MagicMock(ok=False, stdout="", stderr="not found"),
            MagicMock(ok=True, stdout="10.5 0.001\n20.3 0.002\n", stderr=""),
        ]
        result = collector.trace_tcp_latency(duration_sec=1)
        assert "avg_latency_us" in result

    def test_parse_histogram(self, collector):
        """Test histogram parsing from BCC output."""
        # BCC histogram format: [low, high)   count |bars|
        output = """    [1, 2)         1  |
    [2, 4)         5  ||
    [4, 8)        15  |||
"""
        hist = collector._parse_bcc_histogram(output)
        assert len(hist.buckets) == 3
        assert hist.buckets[0].count == 1
        assert hist.buckets[1].count == 5
        assert hist.buckets[0].low == 1
        assert hist.buckets[0].high == 2

    def test_histogram_avg_calculation(self):
        """Test LatencyHistogram avg calculation."""
        from arkprobe.collectors.ebpf_collector import HistogramBucket, LatencyHistogram
        hist = LatencyHistogram(
            unit="us",
            buckets=[
                HistogramBucket(low=0, high=10, count=9),
                HistogramBucket(low=10, high=20, count=1),
            ],
        )
        # avg = (5*9 + 15*1) / 10 = (45+15)/10 = 6.0
        assert hist.avg == 6.0

    def test_histogram_p99_calculation(self):
        """Test LatencyHistogram p99 calculation."""
        from arkprobe.collectors.ebpf_collector import HistogramBucket, LatencyHistogram
        # 100 samples: 99 in bucket 1, 1 in bucket 2
        hist = LatencyHistogram(
            unit="us",
            buckets=[
                HistogramBucket(low=0, high=10, count=99),
                HistogramBucket(low=10, high=20, count=1),
            ],
        )
        assert hist.p99 == 10  # Should return the high of bucket containing p99

    def test_trace_mem_access(self, collector):
        """Test trace_mem_access parses bpftrace output correctly."""
        mock_output = """
@page_faults: 5000
@mmap_calls: 120
@mmap_anon: 100
@mprotect_calls: 30
@brk_calls: 80
@elapsed: 10
"""
        with patch("arkprobe.collectors.ebpf_collector.run_cmd") as mock_run:
            mock_run.return_value = MagicMock(ok=True, stdout=mock_output, stderr="")
            result = collector.trace_mem_access(duration_sec=10)
            assert "page_faults_per_sec" in result
            assert "mmap_calls_per_sec" in result
            assert "anonymous_mmap_ratio" in result
            assert "access_pattern" in result
            assert result["page_faults_per_sec"] == 500.0
            assert result["mmap_calls_per_sec"] == 12.0
            assert result["anonymous_mmap_ratio"] == pytest.approx(0.8333, abs=0.01)

    def test_trace_sched_latency(self, collector):
        """Test trace_sched_latency parses runqlat + runqlen output."""
        mock_histogram_output = """    [0, 1)         100  ||
    [1, 2)         200  |||||
    [2, 4)         150  |||||
    [4, 8)          50  ||
"""
        mock_len_output = """    avg = 2.5
"""
        with patch("arkprobe.collectors.ebpf_collector.run_cmd") as mock_run_cmd:
            mock_run_cmd.side_effect = [
                MagicMock(ok=True, stdout=mock_histogram_output, stderr=""),
                MagicMock(ok=True, stdout=mock_len_output, stderr=""),
            ]
            result = collector.trace_sched_latency(duration_sec=10)
            assert "avg_sched_latency_us" in result
            assert "p99_sched_latency_us" in result
            assert "avg_runqlen" in result
            assert "histogram" in result
            assert "unit" in result
            assert result["avg_runqlen"] == 2.5

    def test_trace_sched_latency_runqlen_fallback(self, collector):
        """Test trace_sched_latency falls back when avg= not found."""
        mock_histogram_output = """    [0, 1)         50  |
"""
        mock_len_output = """runqlen  3
runqlen  4
"""
        with patch("arkprobe.collectors.ebpf_collector.run_cmd") as mock_run_cmd:
            mock_run_cmd.side_effect = [
                MagicMock(ok=True, stdout=mock_histogram_output, stderr=""),
                MagicMock(ok=True, stdout=mock_len_output, stderr=""),
            ]
            result = collector.trace_sched_latency(duration_sec=10)
            assert result["avg_runqlen"] == 3.0