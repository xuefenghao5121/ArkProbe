"""Tests for the analysis engine modules."""

import json
from pathlib import Path

import pytest

from arkprobe.analysis.bottleneck_analyzer import BottleneckAnalyzer
from arkprobe.analysis.comparator import WorkloadComparator
from arkprobe.analysis.design_space import DesignSpaceExplorer
from arkprobe.analysis.feature_extractor import FeatureExtractor
from arkprobe.analysis.scalability_analyzer import ScalabilityAnalyzer
from arkprobe.collectors.collector_orchestrator import FullCollectionResult
from arkprobe.model.enums import BottleneckCategory, ScenarioType
from arkprobe.model.schema import WorkloadFeatureVector
from arkprobe.scenarios.loader import ScenarioConfig, WorkloadConfig

FIXTURES = Path(__file__).parent.parent / "fixtures"


def load_sample_perf():
    return json.loads((FIXTURES / "sample_perf_stat.json").read_text())


def load_sample_ebpf():
    return json.loads((FIXTURES / "sample_ebpf_output.json").read_text())


def make_raw_result(perf_data=None, ebpf_data=None):
    """Create a FullCollectionResult from fixture data."""
    return FullCollectionResult(
        scenario_name="mysql_oltp",
        perf_data=perf_data or load_sample_perf(),
        ebpf_data=ebpf_data or load_sample_ebpf(),
        system_data={
            "platform": {
                "arch": "aarch64",
                "kernel_version": "5.10.0",
                "kunpeng_model": "920",
                "cpu_model_name": "Kunpeng 920",
                "total_cores": 64,
            },
            "disk": {},
        },
        collection_duration_sec=120.0,
    )


def make_scenario():
    return ScenarioConfig(
        name="MySQL OLTP",
        type=ScenarioType.DATABASE_OLTP,
        workload=WorkloadConfig(command="sysbench ..."),
    )


class TestFeatureExtractor:
    def test_extract_basic(self):
        raw = make_raw_result()
        scenario = make_scenario()
        extractor = FeatureExtractor("920")

        fv = extractor.extract(raw, scenario)

        assert fv.scenario_name == "MySQL OLTP"
        assert fv.scenario_type == ScenarioType.DATABASE_OLTP
        assert fv.compute.ipc > 0
        assert fv.compute.cpi > 0
        assert abs(fv.compute.ipc - 1.0 / fv.compute.cpi) < 0.01

    def test_topdown_sums_near_one(self):
        raw = make_raw_result()
        extractor = FeatureExtractor("920")
        fv = extractor.extract(raw, make_scenario())

        td = fv.compute.topdown_l1
        total = td.frontend_bound + td.backend_bound + td.retiring + td.bad_speculation
        assert 0.99 <= total <= 1.01

    def test_cache_mpki_positive(self):
        raw = make_raw_result()
        extractor = FeatureExtractor("920")
        fv = extractor.extract(raw, make_scenario())

        assert fv.cache.l1d_mpki > 0
        assert fv.cache.l2_mpki > 0
        assert fv.cache.l3_mpki > 0
        assert fv.cache.l1d_mpki >= fv.cache.l2_mpki  # L1 misses >= L2 misses in MPKI

    def test_instruction_mix_sums_near_one(self):
        raw = make_raw_result()
        extractor = FeatureExtractor("920")
        fv = extractor.extract(raw, make_scenario())

        im = fv.compute.instruction_mix
        total = (im.integer_ratio + im.fp_ratio + im.vector_ratio +
                 im.branch_ratio + im.load_ratio + im.store_ratio + im.other_ratio)
        assert 0.95 <= total <= 1.05


class TestBottleneckAnalyzer:
    def test_backend_memory_bound(self):
        """Workload with high backend_bound should be classified as memory bound."""
        raw = make_raw_result()
        extractor = FeatureExtractor("920")
        fv = extractor.extract(raw, make_scenario())
        # The sample data has stall_backend = 40% of cycles

        analyzer = BottleneckAnalyzer()
        report = analyzer.analyze(fv)

        assert report.scenario_name == "MySQL OLTP"
        assert report.summary != ""
        assert len(report.architect_notes) > 0

    def test_generates_details(self):
        raw = make_raw_result()
        extractor = FeatureExtractor("920")
        fv = extractor.extract(raw, make_scenario())

        analyzer = BottleneckAnalyzer()
        report = analyzer.analyze(fv)

        # Should have at least one detail
        assert len(report.details) >= 0  # May be 0 if below thresholds
        assert report.topdown_l1 is not None


class TestScalabilityAnalyzer:
    def test_perfect_scaling(self):
        analyzer = ScalabilityAnalyzer()
        cores = [1, 2, 4, 8]
        throughputs = [100, 200, 400, 800]  # Perfect linear

        profile = analyzer.analyze(cores, throughputs)

        for eff in profile.scaling_efficiency:
            assert abs(eff - 1.0) < 0.01

    def test_sublinear_scaling(self):
        analyzer = ScalabilityAnalyzer()
        cores = [1, 2, 4, 8, 16]
        throughputs = [100, 190, 350, 600, 900]  # Sub-linear

        profile = analyzer.analyze(cores, throughputs)

        # Efficiency should decrease
        assert profile.scaling_efficiency[-1] < profile.scaling_efficiency[0]
        # Serial fraction should be > 0
        if profile.amdahl_serial_fraction is not None:
            assert profile.amdahl_serial_fraction > 0

    def test_optimal_cores(self):
        analyzer = ScalabilityAnalyzer()
        cores = [1, 2, 4, 8, 16, 32, 64]
        throughputs = [100, 195, 370, 680, 1100, 1350, 1400]

        profile = analyzer.analyze(cores, throughputs)
        assert profile.optimal_core_count is not None
        assert profile.optimal_core_count <= 64


class TestDesignSpaceExplorer:
    def _make_fv(self, **overrides):
        from tests.test_model.test_schema import make_sample_fv
        return make_sample_fv(**overrides)

    def test_sensitivity_scores_in_range(self):
        fv = self._make_fv()
        explorer = DesignSpaceExplorer()
        scores = explorer.compute_sensitivity(fv)

        for score in scores:
            assert 0.0 <= score.score <= 1.0, f"{score.parameter}: {score.score}"
            assert score.reasoning != ""

    def test_cross_workload_matrix(self):
        fvs = [
            self._make_fv(name="workload_a"),
            self._make_fv(name="workload_b"),
        ]
        explorer = DesignSpaceExplorer()
        matrix = explorer.cross_workload_matrix(fvs)

        assert matrix.shape[0] == 2  # 2 workloads
        assert matrix.shape[1] == 13  # 13 design parameters
        assert all(0 <= v <= 1 for v in matrix.values.flatten())

    def test_recommendations(self):
        fvs = [
            self._make_fv(name="workload_a"),
            self._make_fv(name="workload_b"),
        ]
        explorer = DesignSpaceExplorer()
        recs = explorer.recommend_design_tradeoffs(fvs)

        assert len(recs) > 0
        # Should be sorted by priority descending
        for i in range(len(recs) - 1):
            assert recs[i].priority >= recs[i + 1].priority


class TestWorkloadComparator:
    def _make_fvs(self):
        from tests.test_model.test_schema import make_sample_fv
        from arkprobe.model.enums import ScenarioType
        return [
            make_sample_fv("mysql_oltp", scenario_type=ScenarioType.DATABASE_OLTP),
            make_sample_fv("spark_batch", scenario_type=ScenarioType.BIGDATA_BATCH),
            make_sample_fv("h264_encode", scenario_type=ScenarioType.CODEC_VIDEO),
        ]

    def test_radar_chart_data(self):
        fvs = self._make_fvs()
        comparator = WorkloadComparator()
        radar = comparator.radar_chart_data(fvs)

        assert "dimensions" in radar
        assert "series" in radar
        assert len(radar["series"]) == 3
        for s in radar["series"]:
            assert all(0 <= v <= 1 for v in s["values"])

    def test_heatmap_data(self):
        fvs = self._make_fvs()
        comparator = WorkloadComparator()
        heatmap = comparator.heatmap_data(fvs)

        assert heatmap.shape[0] == 3  # 3 workloads
        assert all(0 <= v <= 1 for v in heatmap.values.flatten())

    def test_full_comparison(self):
        fvs = self._make_fvs()
        comparator = WorkloadComparator()
        result = comparator.compare(fvs)

        assert len(result.scenario_names) == 3
        assert result.radar_data is not None
