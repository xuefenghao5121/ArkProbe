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
        assert collector.BCC_TOOLS is not None
        assert len(collector.BCC_TOOLS) == 5

    def test_bcc_tools_list(self):
        """Verify BCC tools list is correct."""
        expected = [
            "biolatency-bpfcc",
            "cachestat-bpfcc",
            "tcprtt-bpfcc",
            "tcpconnlat-bpfcc",
            "offcputime-bpfcc",
        ]
        assert EbpfCollector.BCC_TOOLS == expected

    @patch("arkprobe.collectors.ebpf_collector.shutil.which")
    def test_is_available_all_tools_present(self, mock_which, collector):
        """Test is_available returns True when all tools present."""
        collector._bcc_available = None  # Reset cache
        collector._bpftrace_available = None
        mock_which.return_value = "/usr/bin/tool"
        available, reason = collector.is_available()
        assert available is True
        assert reason == ""

    @patch("arkprobe.collectors.ebpf_collector.shutil.which")
    @patch("arkprobe.collectors.ebpf_collector.run_cmd")
    def test_is_available_bpftrace_fallback(self, mock_run_cmd, mock_which, collector):
        """Test is_available returns True with bpftrace fallback."""
        collector._bcc_available = None  # Reset cache
        collector._bpftrace_available = None
        # First 5 calls: BCC tools not available
        # Sixth call: bpftrace available
        mock_which.side_effect = [
            None, None, None, None, None,
            "/usr/bin/bpftrace",  # bpftrace
        ]
        mock_run_cmd.return_value = MagicMock(ok=True)
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