"""Tests for the platform optimization analyzer."""

import pytest

from arkprobe.model.enums import ScenarioType, TuningLayer
from arkprobe.model.schema import (
    OSConfig,
    BIOSConfig,
    DriverConfig,
    PlatformConfigSnapshot,
)
from arkprobe.analysis.optimization_analyzer import OptimizationAnalyzer

# Reuse the sample FV builder from test_schema
from tests.test_model.test_schema import make_sample_fv


def _make_config(**os_overrides) -> PlatformConfigSnapshot:
    """Create a PlatformConfigSnapshot with known values."""
    os_cfg = OSConfig(
        hugepages_total=0,
        transparent_hugepage="always",
        cpu_governor="powersave",
        swappiness=60,
        dirty_ratio=20,
        dirty_background_ratio=10,
        numa_balancing=True,
        netdev_max_backlog=1000,
        somaxconn=4096,
        tcp_max_syn_backlog=1024,
        io_schedulers={"sda": "mq-deadline"},
    )
    for k, v in os_overrides.items():
        setattr(os_cfg, k, v)

    return PlatformConfigSnapshot(
        os=os_cfg,
        bios=BIOSConfig(
            numa_enabled=True,
            smt_enabled=True,
            power_profile="powersave",
            c_states_enabled=True,
        ),
        driver=DriverConfig(irqbalance_active=True),
    )


class TestOptimizationAnalyzer:
    def test_all_impact_scores_in_range(self):
        fv = make_sample_fv()
        analyzer = OptimizationAnalyzer()
        report = analyzer.analyze(fv)
        for rec in report.all_recommendations:
            assert 0.0 <= rec.impact_score <= 1.0, \
                f"{rec.parameter_name}: impact {rec.impact_score} out of range"

    def test_gap_detection_with_config(self):
        fv = make_sample_fv()
        fv.platform_config = _make_config(swappiness=60)
        analyzer = OptimizationAnalyzer()
        report = analyzer.analyze(fv)
        swap_rec = next(
            r for r in report.all_recommendations
            if r.parameter_name == "vm.swappiness"
        )
        # Default scenario is DATABASE_OLTP, recommended=1, current=60 → gap
        assert swap_rec.gap_detected is True

    def test_no_gap_when_optimal(self):
        fv = make_sample_fv()
        fv.platform_config = _make_config(swappiness=1)
        analyzer = OptimizationAnalyzer()
        report = analyzer.analyze(fv)
        swap_rec = next(
            r for r in report.all_recommendations
            if r.parameter_name == "vm.swappiness"
        )
        assert swap_rec.gap_detected is False

    def test_scenario_differentiation(self):
        fv_db = make_sample_fv("db", scenario_type=ScenarioType.DATABASE_OLTP)
        fv_ms = make_sample_fv("ms", scenario_type=ScenarioType.MICROSERVICE)
        analyzer = OptimizationAnalyzer()
        report_db = analyzer.analyze(fv_db)
        report_ms = analyzer.analyze(fv_ms)

        swap_db = next(
            r for r in report_db.all_recommendations
            if r.parameter_name == "vm.swappiness"
        )
        swap_ms = next(
            r for r in report_ms.all_recommendations
            if r.parameter_name == "vm.swappiness"
        )
        assert swap_db.recommended_value != swap_ms.recommended_value

    def test_optimization_score_range(self):
        fv = make_sample_fv()
        fv.platform_config = _make_config()
        analyzer = OptimizationAnalyzer()
        report = analyzer.analyze(fv)
        assert 0 <= report.optimization_score <= 100

    def test_cross_scenario_finds_universal(self):
        fv_db = make_sample_fv("db", scenario_type=ScenarioType.DATABASE_OLTP)
        fv_codec = make_sample_fv("codec", scenario_type=ScenarioType.CODEC_VIDEO)
        analyzer = OptimizationAnalyzer()
        cross = analyzer.cross_scenario_analysis([fv_db, fv_codec])
        # power_profile=performance and hw_prefetcher=True are universal
        assert len(cross.universal_recommendations) > 0

    def test_apply_commands_are_valid(self):
        fv = make_sample_fv()
        analyzer = OptimizationAnalyzer()
        report = analyzer.analyze(fv)
        for rec in report.all_recommendations:
            for cmd in rec.apply_commands:
                assert isinstance(cmd, str)
                assert len(cmd) > 0

    def test_recommendations_sorted_by_priority(self):
        fv = make_sample_fv()
        analyzer = OptimizationAnalyzer()
        report = analyzer.analyze(fv)
        for i in range(len(report.all_recommendations) - 1):
            assert report.all_recommendations[i].priority_score >= \
                   report.all_recommendations[i + 1].priority_score

    def test_layer_summaries(self):
        fv = make_sample_fv()
        analyzer = OptimizationAnalyzer()
        report = analyzer.analyze(fv)
        assert "os" in report.layers
        assert "bios" in report.layers
        assert "driver" in report.layers
        assert report.layers["os"].total_parameters > 0

    def test_cross_scenario_matrix_shape(self):
        fvs = [
            make_sample_fv("db", scenario_type=ScenarioType.DATABASE_OLTP),
            make_sample_fv("codec", scenario_type=ScenarioType.CODEC_VIDEO),
            make_sample_fv("ms", scenario_type=ScenarioType.MICROSERVICE),
        ]
        analyzer = OptimizationAnalyzer()
        cross = analyzer.cross_scenario_analysis(fvs)
        assert cross.parameter_benefit_matrix is not None
        assert cross.parameter_benefit_matrix.shape[0] == 3  # 3 scenarios
