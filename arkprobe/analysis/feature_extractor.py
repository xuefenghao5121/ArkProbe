"""Feature extractor: transforms raw collection data into unified feature vectors.

This is the bridge between the collection phase (perf/eBPF/system raw data)
and the analysis phase (unified WorkloadFeatureVector).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..collectors.arm_events import get_kunpeng_model
from ..collectors.collector_orchestrator import FullCollectionResult
from ..model.enums import AccessPattern, IPCMechanism, ScenarioType
from ..model.schema import (
    BranchBehavior,
    CacheHierarchy,
    ComputeCharacteristics,
    ConcurrencyProfile,
    IOCharacteristics,
    InstructionMix,
    MemorySubsystem,
    NetworkCharacteristics,
    PlatformConfigSnapshot,
    TopDownL1,
    TopDownL2,
    WorkloadFeatureVector,
)
from ..scenarios.loader import ScenarioConfig
from ..utils.units import clamp, miss_rate, mpki

log = logging.getLogger(__name__)


class FeatureExtractor:
    """Transforms raw collection data into the unified WorkloadFeatureVector."""

    def __init__(self, kunpeng_model: str = "920"):
        self.hw = get_kunpeng_model(kunpeng_model)

    def extract(
        self,
        raw: FullCollectionResult,
        scenario: ScenarioConfig,
    ) -> WorkloadFeatureVector:
        """Main extraction pipeline."""
        perf = raw.perf_data
        ebpf = raw.ebpf_data
        system = raw.system_data

        platform_info = system.get("platform", {})
        kernel_version = platform_info.get("kernel_version", "unknown")
        platform_name = f"Kunpeng {self.hw.model}" if platform_info.get(
            "kunpeng_model") else platform_info.get("cpu_model_name", "unknown")

        compute = self._extract_compute(perf)
        cache = self._extract_cache(perf)
        branch = self._extract_branch(perf)
        memory = self._extract_memory(perf, system)
        io = self._extract_io(ebpf, system)
        network = self._extract_network(ebpf, system)
        concurrency = self._extract_concurrency(ebpf, system)
        platform_config = self._extract_platform_config(system)

        # Enrich with eBPF data
        cache = self._enrich_cache_with_ebpf(cache, ebpf)
        memory = self._enrich_memory_with_ebpf(memory, ebpf)

        fv = WorkloadFeatureVector(
            scenario_name=scenario.name,
            scenario_type=scenario.type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            platform=platform_name,
            kernel_version=kernel_version,
            collection_duration_sec=raw.collection_duration_sec,
            compute=compute,
            cache=cache,
            branch=branch,
            memory=memory,
            io=io,
            network=network,
            concurrency=concurrency,
            platform_config=platform_config,
        )

        return fv

    # -----------------------------------------------------------------------
    # Compute characteristics
    # -----------------------------------------------------------------------

    def _extract_compute(self, perf: Dict[str, Any]) -> ComputeCharacteristics:
        td = perf.get("topdown_l1", {})
        im = perf.get("instruction_mix", {})

        cycles = td.get("cycles", 1)
        instructions = td.get("instructions", 0)
        ipc = instructions / cycles if cycles > 0 else 0.0
        cpi = 1.0 / ipc if ipc > 0 else 0.0

        # TopDown L1
        stall_fe = td.get("stall_frontend", 0)
        stall_be = td.get("stall_backend", 0)
        frontend_bound = stall_fe / cycles if cycles > 0 else 0.0
        backend_bound = stall_be / cycles if cycles > 0 else 0.0
        retiring = (ipc / self.hw.dispatch_width)
        bad_speculation = 1.0 - frontend_bound - backend_bound - retiring
        bad_speculation = max(0.0, bad_speculation)

        # Normalize to sum to 1.0
        total = frontend_bound + backend_bound + retiring + bad_speculation
        if total > 0:
            frontend_bound /= total
            backend_bound /= total
            retiring /= total
            bad_speculation /= total

        topdown_l1 = TopDownL1(
            frontend_bound=round(clamp(frontend_bound), 4),
            backend_bound=round(clamp(backend_bound), 4),
            retiring=round(clamp(retiring), 4),
            bad_speculation=round(clamp(bad_speculation), 4),
        )

        # Instruction mix
        inst = im.get("inst_retired", 1)
        br = im.get("br_retired", 0)
        fp = im.get("vfp_spec", 0)
        vec = im.get("ase_spec", 0)
        ld = im.get("ld_spec", 0)
        st = im.get("st_spec", 0)

        branch_ratio = br / inst if inst > 0 else 0.0
        fp_ratio = fp / inst if inst > 0 else 0.0
        vector_ratio = vec / inst if inst > 0 else 0.0
        load_ratio = ld / inst if inst > 0 else 0.0
        store_ratio = st / inst if inst > 0 else 0.0
        integer_ratio = max(0.0, 1.0 - branch_ratio - fp_ratio - vector_ratio
                            - load_ratio - store_ratio)

        instruction_mix = InstructionMix(
            integer_ratio=round(integer_ratio, 4),
            fp_ratio=round(fp_ratio, 4),
            vector_ratio=round(vector_ratio, 4),
            branch_ratio=round(branch_ratio, 4),
            load_ratio=round(load_ratio, 4),
            store_ratio=round(store_ratio, 4),
        )

        simd_util = (fp + vec) / inst if inst > 0 else 0.0

        return ComputeCharacteristics(
            ipc=round(ipc, 3),
            cpi=round(cpi, 3),
            instruction_mix=instruction_mix,
            simd_utilization=round(clamp(simd_util), 4),
            topdown_l1=topdown_l1,
        )

    # -----------------------------------------------------------------------
    # Cache hierarchy
    # -----------------------------------------------------------------------

    def _extract_cache(self, perf: Dict[str, Any]) -> CacheHierarchy:
        l1 = perf.get("cache_l1", {})
        l23 = perf.get("cache_l2_l3", {})

        inst = l1.get("inst_retired", 1) or 1

        l1d_access = l1.get("l1d_cache", 1) or 1
        l1d_refill = l1.get("l1d_cache_refill", 0)
        l1i_access = l1.get("l1i_cache", 1) or 1
        l1i_refill = l1.get("l1i_cache_refill", 0)

        l2_access = l23.get("l2d_cache", 1) or 1
        l2_refill = l23.get("l2d_cache_refill", 0)
        l3_access = l23.get("l3d_cache", 1) or 1
        l3_refill = l23.get("l3d_cache_refill", 0)

        inst_for_l23 = l23.get("inst_retired", inst) or 1

        return CacheHierarchy(
            l1i_mpki=round(mpki(l1i_refill, inst), 2),
            l1d_mpki=round(mpki(l1d_refill, inst), 2),
            l2_mpki=round(mpki(l2_refill, inst_for_l23), 2),
            l3_mpki=round(mpki(l3_refill, inst_for_l23), 2),
            l1d_miss_rate=round(miss_rate(l1d_refill, l1d_access), 4),
            l2_miss_rate=round(miss_rate(l2_refill, l2_access), 4),
            l3_miss_rate=round(miss_rate(l3_refill, l3_access), 4),
        )

    def _enrich_cache_with_ebpf(
        self, cache: CacheHierarchy, ebpf: Dict[str, Any]
    ) -> CacheHierarchy:
        """Add locality scores from eBPF data."""
        cache_stats = ebpf.get("cache_stats", {})
        if cache_stats:
            hit_rate = cache_stats.get("hit_rate", 0.5)
            cache.temporal_locality_score = round(hit_rate, 3)

        # Estimate spatial locality from L1D miss rate
        # Low L1D miss rate + high L2 hit rate = good spatial locality
        if cache.l1d_miss_rate < 0.02 and cache.l2_miss_rate < 0.1:
            cache.spatial_locality_score = 0.8
        elif cache.l1d_miss_rate < 0.05:
            cache.spatial_locality_score = 0.5
        else:
            cache.spatial_locality_score = 0.2

        return cache

    # -----------------------------------------------------------------------
    # Branch behavior
    # -----------------------------------------------------------------------

    def _extract_branch(self, perf: Dict[str, Any]) -> BranchBehavior:
        bp = perf.get("branch_prediction", {})
        inst = bp.get("inst_retired", 1) or 1
        br = bp.get("br_retired", 1) or 1
        br_mis = bp.get("br_mis_pred_retired", 0)
        br_immed = bp.get("br_immed_spec", 0)
        br_indirect = bp.get("br_indirect_spec", 0)
        br_return = bp.get("br_return_spec", 0)

        total_spec = br_immed + br_indirect + br_return

        return BranchBehavior(
            branch_mpki=round(mpki(br_mis, inst), 2),
            branch_mispredict_rate=round(miss_rate(br_mis, br), 4),
            indirect_branch_ratio=round(
                br_indirect / total_spec if total_spec > 0 else 0.0, 4
            ),
            branch_density=round(br / inst if inst > 0 else 0.0, 4),
        )

    # -----------------------------------------------------------------------
    # Memory subsystem
    # -----------------------------------------------------------------------

    def _extract_memory(
        self, perf: Dict[str, Any], system: Dict[str, Any]
    ) -> MemorySubsystem:
        ma = perf.get("memory_access", {})
        inst = ma.get("inst_retired", 1) or 1

        dtlb = ma.get("dtlb_walk", 0)
        itlb = ma.get("itlb_walk", 0)
        tlb_mpki_val = mpki(dtlb + itlb, inst)

        # Memory bandwidth - placeholder (will be enriched from uncore or eBPF)
        return MemorySubsystem(
            bandwidth_read_gbps=0.0,
            bandwidth_write_gbps=0.0,
            bandwidth_utilization=0.0,
            tlb_mpki=round(tlb_mpki_val, 2),
        )

    def _enrich_memory_with_ebpf(
        self, memory: MemorySubsystem, ebpf: Dict[str, Any]
    ) -> MemorySubsystem:
        """Enrich memory metrics with eBPF data."""
        # NUMA stats would come from system collector
        return memory

    # -----------------------------------------------------------------------
    # I/O characteristics
    # -----------------------------------------------------------------------

    def _extract_io(
        self, ebpf: Dict[str, Any], system: Dict[str, Any]
    ) -> IOCharacteristics:
        io_data = ebpf.get("io_latency", {})
        disk_data = system.get("disk", {})

        # Aggregate disk stats
        total_iops_r = 0.0
        total_iops_w = 0.0
        total_tp_r = 0.0
        total_tp_w = 0.0
        for dev_stats in disk_data.values():
            if isinstance(dev_stats, dict):
                total_iops_r += dev_stats.get("iops_read", 0)
                total_iops_w += dev_stats.get("iops_write", 0)
                total_tp_r += dev_stats.get("throughput_read_mbps", 0)
                total_tp_w += dev_stats.get("throughput_write_mbps", 0)

        total_iops = total_iops_r + total_iops_w
        rw_ratio = total_iops_r / total_iops if total_iops > 0 else 0.5

        return IOCharacteristics(
            iops_read=round(total_iops_r, 1),
            iops_write=round(total_iops_w, 1),
            throughput_read_mbps=round(total_tp_r, 2),
            throughput_write_mbps=round(total_tp_w, 2),
            avg_latency_us=io_data.get("avg_latency_us"),
            p99_latency_us=io_data.get("p99_latency_us"),
            read_write_ratio=round(rw_ratio, 3),
        )

    # -----------------------------------------------------------------------
    # Network
    # -----------------------------------------------------------------------

    def _extract_network(
        self, ebpf: Dict[str, Any], system: Dict[str, Any]
    ) -> NetworkCharacteristics:
        net_stats = ebpf.get("network_stats", {})
        tcp_data = ebpf.get("tcp_latency", {})

        return NetworkCharacteristics(
            packets_per_sec_rx=net_stats.get("packets_per_sec_rx", 0.0),
            packets_per_sec_tx=net_stats.get("packets_per_sec_tx", 0.0),
            bandwidth_rx_mbps=net_stats.get("bandwidth_rx_mbps", 0.0),
            bandwidth_tx_mbps=net_stats.get("bandwidth_tx_mbps", 0.0),
            avg_latency_us=tcp_data.get("avg_latency_us"),
            p99_latency_us=tcp_data.get("p99_latency_us"),
        )

    # -----------------------------------------------------------------------
    # Concurrency
    # -----------------------------------------------------------------------

    def _extract_concurrency(
        self, ebpf: Dict[str, Any], system: Dict[str, Any]
    ) -> ConcurrencyProfile:
        lock_data = ebpf.get("lock_contention", {})
        cpu_util = system.get("cpu_utilization", {})
        platform = system.get("platform", {})

        thread_count = platform.get("total_cores", 1)

        return ConcurrencyProfile(
            thread_count=thread_count,
            context_switches_per_sec=0.0,  # Filled by system collector
            lock_contention_pct=lock_data.get("lock_contention_pct"),
            futex_wait_time_us=lock_data.get("avg_wait_ns", 0) / 1000
                               if lock_data.get("avg_wait_ns") is not None else None,
        )

    # -----------------------------------------------------------------------
    # Platform configuration
    # -----------------------------------------------------------------------

    def _extract_platform_config(
        self, system: Dict[str, Any]
    ) -> Optional[PlatformConfigSnapshot]:
        """Build PlatformConfigSnapshot from raw system data."""
        config_data = system.get("platform_config")
        if config_data is None:
            return None
        try:
            return PlatformConfigSnapshot.model_validate(config_data)
        except Exception as e:
            log.warning("Failed to parse platform config: %s", e)
            return None
