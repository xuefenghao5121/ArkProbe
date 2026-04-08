"""Tests for the unified feature vector model."""

import json
from pathlib import Path

import pytest

from arkprobe.model.enums import ScenarioType, BottleneckCategory, AccessPattern
from arkprobe.model.schema import (
    BranchBehavior,
    CacheHierarchy,
    ComputeCharacteristics,
    ConcurrencyProfile,
    IOCharacteristics,
    InstructionMix,
    MemorySubsystem,
    NetworkCharacteristics,
    TopDownL1,
    WorkloadFeatureVector,
)
from arkprobe.model.feature_vector import save_feature_vector, load_feature_vector


def make_sample_fv(name: str = "test_workload", **overrides) -> WorkloadFeatureVector:
    """Create a sample feature vector for testing."""
    defaults = dict(
        scenario_name=name,
        scenario_type=ScenarioType.DATABASE_OLTP,
        timestamp="2026-04-01T00:00:00+00:00",
        platform="Kunpeng 920",
        kernel_version="5.10.0",
        collection_duration_sec=120.0,
        compute=ComputeCharacteristics(
            ipc=0.85,
            cpi=1.18,
            instruction_mix=InstructionMix(
                integer_ratio=0.45, fp_ratio=0.01, vector_ratio=0.005,
                branch_ratio=0.15, load_ratio=0.30, store_ratio=0.15,
            ),
            simd_utilization=0.015,
            topdown_l1=TopDownL1(
                frontend_bound=0.15, backend_bound=0.40,
                retiring=0.21, bad_speculation=0.24,
            ),
        ),
        cache=CacheHierarchy(
            l1i_mpki=2.0, l1d_mpki=11.8, l2_mpki=3.5, l3_mpki=1.0,
            l1d_miss_rate=0.02, l2_miss_rate=0.30, l3_miss_rate=0.28,
        ),
        branch=BranchBehavior(
            branch_mpki=6.0, branch_mispredict_rate=0.04,
            indirect_branch_ratio=0.20, branch_density=0.15,
        ),
        memory=MemorySubsystem(
            bandwidth_read_gbps=10.5, bandwidth_write_gbps=5.2,
            bandwidth_utilization=0.08,
        ),
        io=IOCharacteristics(iops_read=5000, iops_write=3000),
        network=NetworkCharacteristics(
            packets_per_sec_rx=125000, packets_per_sec_tx=118000,
        ),
        concurrency=ConcurrencyProfile(thread_count=64),
    )
    defaults.update(overrides)
    return WorkloadFeatureVector(**defaults)


class TestWorkloadFeatureVector:
    def test_create_basic(self):
        fv = make_sample_fv()
        assert fv.scenario_name == "test_workload"
        assert fv.compute.ipc == 0.85
        assert fv.cache.l3_mpki == 1.0

    def test_topdown_sums_to_one(self):
        fv = make_sample_fv()
        td = fv.compute.topdown_l1
        total = td.frontend_bound + td.backend_bound + td.retiring + td.bad_speculation
        assert abs(total - 1.0) < 0.01

    def test_json_roundtrip(self, tmp_path):
        fv = make_sample_fv()
        path = tmp_path / "test_fv.json"
        save_feature_vector(fv, path)

        loaded = load_feature_vector(path)
        assert loaded.scenario_name == fv.scenario_name
        assert loaded.compute.ipc == fv.compute.ipc
        assert loaded.cache.l3_mpki == fv.cache.l3_mpki
        assert loaded.branch.branch_mpki == fv.branch.branch_mpki

    def test_optional_fields(self):
        fv = make_sample_fv()
        assert fv.scalability is None
        assert fv.bottleneck_summary is None
        assert fv.design_sensitivity is None

    def test_design_sensitivity_dict(self):
        fv = make_sample_fv(design_sensitivity={
            "l3_cache_size": 0.82,
            "memory_bandwidth": 0.35,
        })
        assert fv.design_sensitivity["l3_cache_size"] == 0.82


class TestEnums:
    def test_scenario_types(self):
        assert ScenarioType.DATABASE_OLTP.value == "database_oltp"
        assert ScenarioType("codec_video") == ScenarioType.CODEC_VIDEO

    def test_bottleneck_categories(self):
        assert BottleneckCategory.BACKEND_MEMORY_BOUND.value == "backend_memory_bound"
