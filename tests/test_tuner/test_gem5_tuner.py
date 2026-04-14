"""Tests for the gem5 tuner module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from arkprobe.tuner.gem5_tuner import (
    Gem5Tuner,
    Gem5Config,
    Gem5Stats,
    O3CPUConfig,
    CacheConfig,
    GEM5_PRESETS,
)


class TestCacheConfig:
    def test_default_cache(self):
        cache = CacheConfig()
        assert cache.size_kb == 64
        assert cache.assoc == 4

    def test_to_gem5_size_kb(self):
        cache = CacheConfig(size_kb=32)
        assert cache.to_gem5_size() == "32KiB"

    def test_to_gem5_size_mb(self):
        cache = CacheConfig(size_kb=2048)
        assert cache.to_gem5_size() == "2MiB"


class TestO3CPUConfig:
    def test_default_config(self):
        config = O3CPUConfig()
        assert config.fetch_width == 4
        assert config.rob_entries == 128
        assert config.btb_entries == 2048

    def test_custom_config(self):
        config = O3CPUConfig(
            fetch_width=8,
            rob_entries=256,
            btb_entries=4096,
        )
        assert config.fetch_width == 8
        assert config.rob_entries == 256


class TestGem5Config:
    def test_default_config(self):
        config = Gem5Config()
        assert config.name == "default"
        assert config.cpu_freq == "3GHz"
        assert config.l2_cache is not None

    def test_to_dict(self):
        config = Gem5Config(
            name="test",
            cpu_config=O3CPUConfig(rob_entries=256),
            l1i_cache=CacheConfig(size_kb=32),
        )
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["cpu_config"]["rob_entries"] == 256
        assert d["l1i_cache"]["size_kb"] == 32

    def test_no_l2_cache(self):
        config = Gem5Config(l2_cache=None)
        assert config.to_dict()["l2_cache"] is None


class TestGem5Stats:
    def test_default_stats(self):
        stats = Gem5Stats()
        assert stats.ipc == 0.0
        assert stats.instructions == 0

    def test_ipc_calculation(self):
        stats = Gem5Stats(instructions=1000, cycles=500)
        # IPC would be calculated externally
        assert stats.instructions == 1000
        assert stats.cycles == 500


class TestGem5Presets:
    def test_presets_exist(self):
        assert "default" in GEM5_PRESETS
        assert "small_cache" in GEM5_PRESETS
        assert "large_cache" in GEM5_PRESETS
        assert "wide_issue" in GEM5_PRESETS
        assert "deep_rob" in GEM5_PRESETS
        assert "kunpeng_like" in GEM5_PRESETS

    def test_small_cache_preset(self):
        config = GEM5_PRESETS["small_cache"]
        assert config.l1i_cache.size_kb < 32
        assert config.l1d_cache.size_kb < 32

    def test_wide_issue_preset(self):
        config = GEM5_PRESETS["wide_issue"]
        assert config.cpu_config.issue_width == 8

    def test_deep_rob_preset(self):
        config = GEM5_PRESETS["deep_rob"]
        assert config.cpu_config.rob_entries > 128

    def test_kunpeng_like_preset(self):
        config = GEM5_PRESETS["kunpeng_like"]
        assert config.cpu_config.issue_width == 4
        assert config.l1i_cache.size_kb == 64


class TestGem5Tuner:
    @pytest.fixture
    def tuner_no_gem5(self):
        """Tuner without gem5 installed."""
        return Gem5Tuner(gem5_path=None)

    def test_no_gem5_available(self, tuner_no_gem5):
        assert tuner_no_gem5.gem5_path is None

    def test_generate_config_script(self, tuner_no_gem5):
        config = Gem5Config(name="test_config")
        script_path = tuner_no_gem5.generate_config_script(config)

        assert script_path.exists()
        content = script_path.read_text()
        assert "test_config" in content
        assert "O3_ARM_CPU" in content

    def test_config_script_has_cache_params(self, tuner_no_gem5):
        config = Gem5Config(
            name="cache_test",
            l1i_cache=CacheConfig(size_kb=64, assoc=4),
            l1d_cache=CacheConfig(size_kb=64, assoc=4),
        )
        script_path = tuner_no_gem5.generate_config_script(config)
        content = script_path.read_text()

        assert "64KiB" in content
        assert "assoc = 4" in content

    def test_config_script_has_cpu_params(self, tuner_no_gem5):
        config = Gem5Config(
            name="cpu_test",
            cpu_config=O3CPUConfig(
                fetch_width=8,
                rob_entries=256,
            ),
        )
        script_path = tuner_no_gem5.generate_config_script(config)
        content = script_path.read_text()

        assert "fetchWidth = 8" in content
        assert "numROBEntries = 256" in content

    def test_stats_to_feature_dict(self, tuner_no_gem5):
        stats = Gem5Stats(
            ipc=1.5,
            instructions=1000000,
            cycles=666666,
            l1d_mpki=10.0,
            branch_mpki=5.0,
        )
        d = tuner_no_gem5.stats_to_feature_dict(stats)

        assert d["ipc"] == 1.5
        assert d["instructions"] == 1000000
        assert d["l1d_mpki"] == 10.0

    def test_simulate_without_gem5_raises(self, tuner_no_gem5):
        with pytest.raises(RuntimeError, match="gem5 not available"):
            tuner_no_gem5.simulate(Gem5Config(), Path("/tmp/binary"))

    def test_parse_stats(self, tuner_no_gem5, tmp_path):
        stats_content = """simTicks: 1000000000
simSeconds: 0.001
system.cpu.committedInsts: 1000000
system.cpu.numCycles: 500000
system.cpu.icache.demandAccesses::total: 500000
system.cpu.icache.demandMisses::total: 5000
system.cpu.dcache.demandAccesses::total: 400000
system.cpu.dcache.demandMisses::total: 8000
system.cpu.branchPred.lookups: 100000
system.cpu.branchPred.mispredicted: 2000
"""
        stats_file = tmp_path / "stats.txt"
        stats_file.write_text(stats_content)

        stats = tuner_no_gem5.parse_stats(stats_file)

        assert stats.sim_ticks == 1000000000
        assert stats.instructions == 1000000
        assert stats.cycles == 500000
        assert stats.ipc == 2.0  # 1M / 500K
        assert stats.l1i_misses == 5000
        assert stats.l1d_misses == 8000
        assert stats.branch_mispredicted == 2000


class TestGem5ConfigScript:
    """Test generated gem5 configuration scripts."""

    @pytest.fixture
    def tuner_no_gem5(self):
        return Gem5Tuner(gem5_path=None)

    def test_script_with_l2_cache(self, tuner_no_gem5):
        config = Gem5Config(
            name="with_l2",
            l2_cache=CacheConfig(size_kb=512),
        )
        script_path = tuner_no_gem5.generate_config_script(config)
        content = script_path.read_text()

        assert "class L2Cache" in content
        assert "512KiB" in content
        assert "L2XBar" in content

    def test_script_without_l2_cache(self, tuner_no_gem5):
        config = Gem5Config(
            name="no_l2",
            l2_cache=None,
        )
        script_path = tuner_no_gem5.generate_config_script(config)
        content = script_path.read_text()

        # Should connect L1 directly to memory bus
        assert "system.cpu.icache.mem_side = system.membus" in content

    def test_script_has_memory_config(self, tuner_no_gem5):
        config = Gem5Config(
            name="mem_test",
            mem_size="4GB",
            mem_type="DDR4_2400_8x8",
        )
        script_path = tuner_no_gem5.generate_config_script(config)
        content = script_path.read_text()

        assert "4GB" in content
        assert "DDR4_2400_8x8" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
