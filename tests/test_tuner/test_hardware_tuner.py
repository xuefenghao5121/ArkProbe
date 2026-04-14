"""Tests for the hardware tuner module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from arkprobe.tuner.hardware_tuner import (
    HardwareTuner,
    TuningConfig,
    TuningResult,
    SystemState,
    CPUGovernor,
    CStateLimit,
    NUMAPolicy,
    THPSetting,
    TUNING_PRESETS,
)
from arkprobe.tuner.comparator import (
    TuningComparator,
    ImpactReport,
    MetricChange,
    BottleneckShift,
)


class TestTuningConfig:
    def test_default_config(self):
        config = TuningConfig()
        assert config.name == "default"
        assert config.cpu_governor == CPUGovernor.PERFORMANCE
        assert config.smt_enabled is True
        assert config.cstate_limit == CStateLimit.UNLIMITED

    def test_to_dict_and_from_dict(self):
        config = TuningConfig(
            name="test",
            cpu_governor=CPUGovernor.POWERSAVE,
            cpu_frequency_mhz=2600,
            smt_enabled=False,
            cstate_limit=CStateLimit.C1_MAX,
            numa_policy=NUMAPolicy.BIND,
            numa_nodes=[0],
            cpu_affinity=[0, 1, 2, 3],
            thp_setting=THPSetting.ALWAYS,
            description="Test config",
        )

        d = config.to_dict()
        restored = TuningConfig.from_dict(d)

        assert restored.name == config.name
        assert restored.cpu_governor == config.cpu_governor
        assert restored.cpu_frequency_mhz == config.cpu_frequency_mhz
        assert restored.smt_enabled == config.smt_enabled
        assert restored.cstate_limit == config.cstate_limit
        assert restored.numa_policy == config.numa_policy
        assert restored.numa_nodes == config.numa_nodes
        assert restored.cpu_affinity == config.cpu_affinity
        assert restored.thp_setting == config.thp_setting

    def test_presets_exist(self):
        assert "default" in TUNING_PRESETS
        assert "performance" in TUNING_PRESETS
        assert "latency" in TUNING_PRESETS
        assert "power" in TUNING_PRESETS
        assert "database" in TUNING_PRESETS
        assert "compute" in TUNING_PRESETS

    def test_performance_preset(self):
        config = TUNING_PRESETS["performance"]
        assert config.cpu_governor == CPUGovernor.PERFORMANCE
        assert config.smt_enabled is True
        assert config.cstate_limit == CStateLimit.C1_MAX

    def test_database_preset_no_smt(self):
        config = TUNING_PRESETS["database"]
        assert config.smt_enabled is False


class TestHardwareTuner:
    @pytest.fixture
    def tuner(self):
        return HardwareTuner(dry_run=True)

    def test_dry_run_mode(self, tuner):
        assert tuner.dry_run is True

    def test_get_numactl_cmd_default(self, tuner):
        config = TuningConfig(numa_policy=NUMAPolicy.DEFAULT)
        cmd = tuner.get_numactl_cmd(config)
        assert cmd == []

    def test_get_numactl_cmd_bind(self, tuner):
        config = TuningConfig(
            numa_policy=NUMAPolicy.BIND,
            numa_nodes=[0, 1],
        )
        cmd = tuner.get_numactl_cmd(config)
        assert "numactl" in cmd
        assert "--cpunodebind=0,1" in cmd
        assert "--membind=0,1" in cmd

    def test_get_numactl_cmd_interleave(self, tuner):
        config = TuningConfig(
            numa_policy=NUMAPolicy.INTERLEAVE,
            numa_nodes=[0, 1],
        )
        cmd = tuner.get_numactl_cmd(config)
        assert "numactl" in cmd
        assert "--interleave=0,1" in cmd

    def test_get_taskset_cmd_no_affinity(self, tuner):
        config = TuningConfig()
        cmd = tuner.get_taskset_cmd(config)
        assert cmd == []

    def test_get_taskset_cmd_with_affinity(self, tuner):
        config = TuningConfig(cpu_affinity=[0, 2, 4, 6])
        cmd = tuner.get_taskset_cmd(config)
        assert "taskset" in cmd
        # CPU 0, 2, 4, 6 -> mask = 0b01010101 = 0x55
        assert "0x55" in cmd

    def test_wrap_command(self, tuner):
        config = TuningConfig(
            numa_policy=NUMAPolicy.BIND,
            numa_nodes=[0],
            cpu_affinity=[0],
        )
        cmd = ["./workload", "--threads", "4"]
        wrapped = tuner.wrap_command(config, cmd)
        assert "numactl" in wrapped
        assert "taskset" in wrapped
        assert "./workload" in wrapped

    def test_apply_dry_run(self, tuner):
        config = TUNING_PRESETS["performance"]
        result = tuner.apply(config)
        assert result.success is True
        assert result.config == config

    @patch("os.path.exists")
    def test_get_current_state_mock(self, mock_exists, tuner):
        # This is a basic test - real testing needs a real system
        # or more extensive mocking
        pass


class TestMetricChange:
    def test_ipc_improved(self):
        change = MetricChange(
            name="ipc",
            baseline_value=1.0,
            tuned_value=1.2,
            absolute_change=0.2,
            percent_change=20.0,
        )
        assert change.improved is True

    def test_ipc_degraded(self):
        change = MetricChange(
            name="ipc",
            baseline_value=1.2,
            tuned_value=1.0,
            absolute_change=-0.2,
            percent_change=-16.67,
        )
        assert change.improved is False

    def test_mpki_improved(self):
        # Lower MPKI is better
        change = MetricChange(
            name="l3_mpki",
            baseline_value=10.0,
            tuned_value=8.0,
            absolute_change=-2.0,
            percent_change=-20.0,
        )
        assert change.improved is True

    def test_mpki_degraded(self):
        change = MetricChange(
            name="l3_mpki",
            baseline_value=8.0,
            tuned_value=10.0,
            absolute_change=2.0,
            percent_change=25.0,
        )
        assert change.improved is False


class TestTuningComparator:
    @pytest.fixture
    def comparator(self):
        return TuningComparator()

    def _create_mock_fv(self, name: str, ipc: float, l3_mpki: float,
                        branch_mpki: float, backend_bound: float):
        """Create a minimal mock feature vector for testing."""
        from arkprobe.model.schema import (
            WorkloadFeatureVector,
            ComputeCharacteristics,
            CacheHierarchy,
            BranchBehavior,
            MemorySubsystem,
            IOCharacteristics,
            NetworkCharacteristics,
            ConcurrencyProfile,
            ScalabilityProfile,
            PowerThermal,
            TopDownL1,
            InstructionMix,
        )
        from arkprobe.model.enums import ScenarioType
        from datetime import datetime

        return WorkloadFeatureVector(
            scenario_name=name,
            scenario_type=ScenarioType.MICROSERVICE,
            timestamp=datetime.now().isoformat(),
            platform="Kunpeng 920",
            kernel_version="5.10.0",
            collection_duration_sec=60.0,
            compute=ComputeCharacteristics(
                ipc=ipc,
                cpi=1.0 / ipc if ipc > 0 else 1.0,
                instruction_mix=InstructionMix(
                    integer_ratio=0.4,
                    fp_ratio=0.1,
                    vector_ratio=0.1,
                    branch_ratio=0.1,
                    load_ratio=0.2,
                    store_ratio=0.1,
                ),
                topdown_l1=TopDownL1(
                    frontend_bound=0.1,
                    backend_bound=backend_bound,
                    retiring=1.0 - 0.1 - backend_bound - 0.1,
                    bad_speculation=0.1,
                ),
            ),
            cache=CacheHierarchy(
                l1i_mpki=1.0,
                l1d_mpki=2.0,
                l2_mpki=3.0,
                l3_mpki=l3_mpki,
                l1d_miss_rate=0.02,
                l2_miss_rate=0.1,
                l3_miss_rate=0.3,
            ),
            branch=BranchBehavior(
                branch_mpki=branch_mpki,
                branch_mispredict_rate=0.1,
            ),
            memory=MemorySubsystem(
                bandwidth_read_gbps=10.0,
                bandwidth_write_gbps=5.0,
                bandwidth_utilization=0.5,
                tlb_mpki=0.5,
            ),
            io=IOCharacteristics(),
            network=NetworkCharacteristics(),
            concurrency=ConcurrencyProfile(thread_count=4),
            scalability=ScalabilityProfile(
                core_counts=[1, 2, 4],
                throughput_at_core_count=[100, 180, 300],
                scaling_efficiency=[1.0, 0.9, 0.75],
            ),
            power_thermal=PowerThermal(),
        )

    @pytest.fixture
    def mock_baseline_fv(self):
        return self._create_mock_fv("baseline", ipc=1.0, l3_mpki=10.0,
                                    branch_mpki=5.0, backend_bound=0.4)

    @pytest.fixture
    def mock_improved_fv(self):
        return self._create_mock_fv("improved", ipc=1.2, l3_mpki=8.0,
                                    branch_mpki=4.0, backend_bound=0.3)

    def test_compare_basic(self, comparator, mock_baseline_fv, mock_improved_fv):
        report = comparator.compare(mock_baseline_fv, mock_improved_fv)

        assert report.config_name == "improved"
        assert report.baseline_name == "baseline"
        assert len(report.metric_changes) > 0

    def test_compare_ipc_improvement(self, comparator, mock_baseline_fv, mock_improved_fv):
        report = comparator.compare(mock_baseline_fv, mock_improved_fv)

        ipc_change = next((c for c in report.metric_changes if c.name == "ipc"), None)
        assert ipc_change is not None
        assert ipc_change.percent_change > 0
        assert ipc_change.improved is True

    def test_compare_l3_mpki_improvement(self, comparator, mock_baseline_fv, mock_improved_fv):
        report = comparator.compare(mock_baseline_fv, mock_improved_fv)

        l3_change = next((c for c in report.metric_changes if c.name == "l3_mpki"), None)
        assert l3_change is not None
        assert l3_change.percent_change < 0  # Reduced MPKI
        assert l3_change.improved is True

    def test_bottleneck_shift(self, comparator, mock_baseline_fv, mock_improved_fv):
        report = comparator.compare(mock_baseline_fv, mock_improved_fv)

        assert report.bottleneck_shift is not None
        # Baseline is Backend-bound (0.4), improved is still Backend but less severe
        assert report.bottleneck_shift.baseline_bottleneck == "Backend"
        assert report.bottleneck_shift.tuned_bottleneck == "Backend"
        assert report.bottleneck_shift.severity_change < 0  # Reduced

    def test_overall_improvement(self, comparator, mock_baseline_fv, mock_improved_fv):
        report = comparator.compare(mock_baseline_fv, mock_improved_fv)

        # Should show overall improvement
        assert report.overall_improvement > 0

    def test_key_findings(self, comparator, mock_baseline_fv, mock_improved_fv):
        report = comparator.compare(mock_baseline_fv, mock_improved_fv)

        assert len(report.key_findings) > 0
        # Should mention IPC improvement
        assert any("IPC" in f for f in report.key_findings)

    def test_to_dict(self, comparator, mock_baseline_fv, mock_improved_fv):
        report = comparator.compare(mock_baseline_fv, mock_improved_fv)
        d = report.to_dict()

        assert d["config_name"] == "improved"
        assert d["baseline_name"] == "baseline"
        assert "metric_changes" in d
        assert "overall_improvement" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
