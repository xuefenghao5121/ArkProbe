"""Tests for JVM feature vector models and extraction."""

import pytest

from arkprobe.model.enums import (
    BottleneckCategory,
    ScenarioType,
    TuningLayer,
)
from arkprobe.model.schema import (
    GCMetrics,
    JITMetrics,
    JVMThreadMetrics,
    JvmCharacteristics,
    WorkloadFeatureVector,
    ComputeCharacteristics,
    CacheHierarchy,
    BranchBehavior,
    MemorySubsystem,
    IOCharacteristics,
    NetworkCharacteristics,
    ConcurrencyProfile,
    InstructionMix,
    TopDownL1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_fv(**overrides) -> WorkloadFeatureVector:
    defaults = dict(
        scenario_name="jvm_test",
        scenario_type=ScenarioType.JVM_GENERAL,
        timestamp="2026-04-22T00:00:00+00:00",
        platform="Kunpeng 920",
        kernel_version="5.10.0",
        collection_duration_sec=60.0,
        compute=ComputeCharacteristics(
            ipc=1.5,
            cpi=0.67,
            instruction_mix=InstructionMix(
                integer_ratio=0.5, fp_ratio=0.1, vector_ratio=0.05,
                branch_ratio=0.1, load_ratio=0.2, store_ratio=0.05,
            ),
            topdown_l1=TopDownL1(
                frontend_bound=0.1, backend_bound=0.3,
                retiring=0.5, bad_speculation=0.1,
            ),
        ),
        cache=CacheHierarchy(
            l1i_mpki=1.0, l1d_mpki=5.0, l2_mpki=3.0, l3_mpki=2.0,
            l1d_miss_rate=0.02, l2_miss_rate=0.05, l3_miss_rate=0.1,
        ),
        branch=BranchBehavior(branch_mpki=2.0, branch_mispredict_rate=0.03),
        memory=MemorySubsystem(
            bandwidth_read_gbps=10.0, bandwidth_write_gbps=5.0,
            bandwidth_utilization=0.4,
        ),
        io=IOCharacteristics(),
        network=NetworkCharacteristics(),
        concurrency=ConcurrencyProfile(thread_count=64),
    )
    defaults.update(overrides)
    return WorkloadFeatureVector(**defaults)


# ---------------------------------------------------------------------------
# Model construction tests
# ---------------------------------------------------------------------------

class TestGCMetrics:
    def test_defaults(self):
        gc = GCMetrics(gc_algorithm="G1")
        assert gc.gc_algorithm == "G1"
        assert gc.young_gc_count == 0
        assert gc.gc_pause_ratio == 0.0
        assert gc.heap_usage_ratio == 0.0

    def test_with_values(self):
        gc = GCMetrics(
            gc_algorithm="ZGC",
            young_gc_count=100,
            young_gc_total_ms=500.0,
            full_gc_count=2,
            full_gc_total_ms=200.0,
            gc_pause_ratio=0.08,
            avg_gc_pause_ms=6.86,
            max_gc_pause_ms=50.0,
            heap_used_mb=4096.0,
            heap_max_mb=8192.0,
            heap_usage_ratio=0.50,
            metaspace_used_mb=256.0,
        )
        assert gc.gc_algorithm == "ZGC"
        assert gc.gc_pause_ratio == 0.08
        assert gc.heap_usage_ratio == 0.5


class TestJITMetrics:
    def test_defaults(self):
        jit = JITMetrics()
        assert jit.total_compilations == 0
        assert jit.deopt_ratio == 0.0

    def test_with_values(self):
        jit = JITMetrics(
            total_compilations=500,
            compilations_per_sec=8.33,
            deoptimization_count=25,
            deopt_ratio=0.05,
            c1_count=300,
            c2_count=200,
            osr_count=50,
        )
        assert jit.deopt_ratio == 0.05
        assert jit.c1_count + jit.c2_count == 500


class TestJVMThreadMetrics:
    def test_defaults(self):
        tm = JVMThreadMetrics()
        assert tm.total_threads == 0
        assert tm.safepoint_ratio == 0.0

    def test_with_values(self):
        tm = JVMThreadMetrics(
            total_threads=200,
            active_threads=150,
            daemon_threads=50,
            safepoint_count=1000,
            safepoint_total_ms=3000.0,
            safepoint_ratio=0.05,
        )
        assert tm.safepoint_ratio == 0.05


class TestJvmCharacteristics:
    def test_defaults(self):
        jvm = JvmCharacteristics(jdk_version="17.0.6")
        assert jvm.jdk_version == "17.0.6"
        assert jvm.jfr_available is False
        assert jvm.jfr_events_collected == []

    def test_with_jfr_data(self):
        gc = GCMetrics(gc_algorithm="G1", gc_pause_ratio=0.12)
        jit = JITMetrics(total_compilations=500, deopt_ratio=0.03)
        threads = JVMThreadMetrics(total_threads=200, safepoint_ratio=0.02)

        jvm = JvmCharacteristics(
            jdk_version="17.0.6",
            gc=gc,
            jit=jit,
            threads=threads,
            jfr_available=True,
            jfr_events_collected=["gc", "jit", "thread"],
        )
        assert jvm.gc.gc_algorithm == "G1"
        assert jvm.jit.total_compilations == 500
        assert jvm.threads.total_threads == 200
        assert jvm.jfr_available is True


# ---------------------------------------------------------------------------
# Enum extension tests
# ---------------------------------------------------------------------------

class TestJVMEnums:
    def test_scenario_types(self):
        assert ScenarioType.JVM_GENERAL.value == "jvm_general"
        assert ScenarioType.JVM_GC_HEAVY.value == "jvm_gc_heavy"
        assert ScenarioType.JVM_JIT_INTENSIVE.value == "jvm_jit_intensive"

    def test_bottleneck_categories(self):
        assert BottleneckCategory.JVM_GC_PAUSE.value == "jvm_gc_pause"
        assert BottleneckCategory.JVM_SAFEPPOINT.value == "jvm_safepoint"
        assert BottleneckCategory.JVM_JIT_DEOPT.value == "jvm_jit_deopt"
        assert BottleneckCategory.JVM_HEAP_PRESSURE.value == "jvm_heap_pressure"

    def test_tuning_layer(self):
        assert TuningLayer.JVM.value == "jvm"


# ---------------------------------------------------------------------------
# Feature vector integration tests
# ---------------------------------------------------------------------------

class TestWorkloadFeatureVectorWithJVM:
    def test_fv_without_jvm(self):
        fv = _make_base_fv()
        assert fv.jvm is None

    def test_fv_with_jvm(self):
        jvm = JvmCharacteristics(
            jdk_version="17.0.6",
            gc=GCMetrics(gc_algorithm="G1", gc_pause_ratio=0.08),
            jit=JITMetrics(total_compilations=300),
            threads=JVMThreadMetrics(total_threads=100),
            jfr_available=True,
            jfr_events_collected=["gc", "jit", "thread"],
        )
        fv = _make_base_fv(jvm=jvm)
        assert fv.jvm is not None
        assert fv.jvm.gc.gc_algorithm == "G1"
        assert fv.jvm.gc.gc_pause_ratio == 0.08
        assert fv.jvm.jfr_available is True

    def test_fv_serialization_with_jvm(self):
        jvm = JvmCharacteristics(
            jdk_version="11.0.18",
            gc=GCMetrics(gc_algorithm="Parallel"),
        )
        fv = _make_base_fv(jvm=jvm)
        data = fv.model_dump()
        assert data["jvm"]["gc"]["gc_algorithm"] == "Parallel"

        # Round-trip
        fv2 = WorkloadFeatureVector.model_validate(data)
        assert fv2.jvm.gc.gc_algorithm == "Parallel"
