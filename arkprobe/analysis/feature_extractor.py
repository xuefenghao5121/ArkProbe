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
    GCMetrics,
    IOCharacteristics,
    InstructionMix,
    JITMetrics,
    JVMThreadMetrics,
    JvmCharacteristics,
    MemorySubsystem,
    NetworkCharacteristics,
    PlatformConfigSnapshot,
    PowerThermal,
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
        power_thermal = self._extract_power_thermal(system)
        platform_config = self._extract_platform_config(system)

        # Enrich with eBPF data
        cache = self._enrich_cache_with_ebpf(cache, ebpf)
        memory = self._enrich_memory_with_ebpf(memory, ebpf)

        # JVM characteristics (if JFR data available)
        jvm = self._extract_jvm(raw.jvm_data) if raw.jvm_data else None

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
            power_thermal=power_thermal,
            platform_config=platform_config,
            jvm=jvm,
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

        # Instruction mix — 930 lacks vfp_spec/ase_spec/ld_spec/st_spec
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
        known_ratio = branch_ratio + fp_ratio + vector_ratio + load_ratio + store_ratio
        integer_ratio = max(0.0, 1.0 - known_ratio)

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

        # 920: br_immed_spec, br_indirect_spec, br_return_spec
        # 930: br_pred, br_mis_pred, br_return_retired (no spec variants)
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

        # Memory bandwidth from uncore PMU (if available)
        uncore = perf.get("uncore", {})
        ddr_bw = uncore.get("ddr_bandwidth", {})

        read_gbps = ddr_bw.get("read_gbps", 0.0) if ddr_bw else 0.0
        write_gbps = ddr_bw.get("write_gbps", 0.0) if ddr_bw else 0.0
        utilization = ddr_bw.get("utilization", 0.0) if ddr_bw else 0.0

        return MemorySubsystem(
            bandwidth_read_gbps=round(read_gbps, 2),
            bandwidth_write_gbps=round(write_gbps, 2),
            bandwidth_utilization=round(utilization, 4),
            tlb_mpki=round(tlb_mpki_val, 2),
        )

    def _enrich_memory_with_ebpf(
        self, memory: MemorySubsystem, ebpf: Dict[str, Any]
    ) -> MemorySubsystem:
        """Enrich memory metrics with eBPF data."""
        mem_access = ebpf.get("mem_access", {})
        if mem_access:
            pattern_str = mem_access.get("access_pattern")
            if pattern_str:
                from ..model.enums import AccessPattern
                pattern_map = {
                    "sequential": AccessPattern.STREAMING,
                    "random": AccessPattern.RANDOM,
                    "mixed": AccessPattern.MIXED,
                }
                memory.access_pattern = pattern_map.get(pattern_str, AccessPattern.MIXED)
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
        sched_data = ebpf.get("sched_latency", {})
        cpu_util = system.get("cpu_utilization", {})
        platform = system.get("platform", {})

        thread_count = platform.get("total_cores", 1)

        return ConcurrencyProfile(
            thread_count=thread_count,
            context_switches_per_sec=0.0,  # Filled by system collector
            lock_contention_pct=lock_data.get("lock_contention_pct"),
            futex_wait_time_us=lock_data.get("avg_wait_ns", 0) / 1000
                               if lock_data.get("avg_wait_ns") is not None else None,
            avg_sched_latency_us=sched_data.get("avg_sched_latency_us"),
            p99_sched_latency_us=sched_data.get("p99_sched_latency_us"),
        )

    # -----------------------------------------------------------------------
    # Power and thermal characteristics
    # -----------------------------------------------------------------------

    def _extract_power_thermal(
        self, system: Dict[str, Any]
    ) -> Optional[PowerThermal]:
        """Extract power and thermal characteristics from system data."""
        pt_data = system.get("power_thermal")
        if pt_data is None:
            return None

        try:
            return PowerThermal(
                cpu_power_w=pt_data.get("cpu_power_w"),
                gpu_power_w=pt_data.get("gpu_power_w"),
                dram_power_w=pt_data.get("dram_power_w"),
                total_power_w=pt_data.get("total_power_w"),
                cpu_temp_c=pt_data.get("cpu_temp_c"),
                cpu_temp_max_c=pt_data.get("cpu_temp_max_c"),
                dram_temp_c=pt_data.get("dram_temp_c"),
                motherboard_temp_c=pt_data.get("motherboard_temp_c"),
                c0_residency=pt_data.get("c0_residency"),
                c1_residency=pt_data.get("c1_residency"),
                c2_residency=pt_data.get("c2_residency"),
                c3_residency=pt_data.get("c3_residency"),
                c6_residency=pt_data.get("c6_residency"),
                avg_freq_mhz=pt_data.get("avg_freq_mhz"),
                min_freq_mhz=pt_data.get("min_freq_mhz"),
                max_freq_mhz=pt_data.get("max_freq_mhz"),
                thermal_throttling_pct=pt_data.get("thermal_throttling_pct"),
            )
        except Exception as e:
            log.warning("Failed to parse power_thermal: %s", e)
            return None

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

    # -----------------------------------------------------------------------
    # JVM characteristics
    # -----------------------------------------------------------------------

    def _extract_jvm(self, jvm_result: "CollectionResult") -> JvmCharacteristics:
        """Extract JVM characteristics from JFR/jstat collection result."""
        from ..collectors.base import CollectionResult

        data = jvm_result.data
        jdk_version = data.get("jdk_version", "unknown")
        jfr_available = data.get("jfr_available", False)
        events_collected = data.get("jfr_events_collected", [])

        if jfr_available:
            gc = self._extract_gc_from_jfr(data)
            jit = self._extract_jit_from_jfr(data)
            threads = self._extract_threads_from_jfr(data)
        else:
            gc = self._extract_gc_from_jstat(data)
            jit = JITMetrics()  # jstat has no JIT data
            threads = self._extract_threads_from_jstat(data)

        return JvmCharacteristics(
            jdk_version=jdk_version,
            gc=gc,
            jit=jit,
            threads=threads,
            jfr_available=jfr_available,
            jfr_events_collected=events_collected,
        )

    def _extract_gc_from_jfr(self, data: Dict[str, Any]) -> GCMetrics:
        """Extract GC metrics from JFR parsed data."""
        jfr_parsed = data.get("jfr_parsed", {})
        gc_events = jfr_parsed.get("gc_events", [])

        gc_algorithm = "unknown"
        young_gc_count = 0
        young_gc_total_ms = 0.0
        full_gc_count = 0
        full_gc_total_ms = 0.0
        heap_used_mb = 0.0
        heap_max_mb = 0.0
        metaspace_used_mb = 0.0

        for event in gc_events:
            event_type = event.get("type", "")
            props = event.get("values", event)

            if "GCConfiguration" in event_type:
                gc_algorithm = props.get("collectorName", gc_algorithm)
            elif "GCHeapSummary" in event_type:
                heap_used = props.get("heapUsed", 0)
                heap_max_val = props.get("heapSpace", {}).get("size", 0)
                if heap_used > 0:
                    heap_used_mb = heap_used / (1024 * 1024)
                if heap_max_val > 0:
                    heap_max_mb = heap_max_val / (1024 * 1024)
            elif "OldGC" in event_type or "FullGC" in event_type:
                full_gc_count += 1
                duration_ns = props.get("duration", 0)
                if isinstance(duration_ns, (int, float)):
                    full_gc_total_ms += duration_ns / 1_000_000
            elif "YoungGC" in event_type or "GCPhaseLevel" in event_type:
                young_gc_count += 1
                duration_ns = props.get("duration", 0)
                if isinstance(duration_ns, (int, float)):
                    young_gc_total_ms += duration_ns / 1_000_000
            elif "MetaspaceSummary" in event_type:
                ms_used = props.get("metaspace", {}).get("used", 0)
                if isinstance(ms_used, (int, float)):
                    metaspace_used_mb = ms_used / (1024 * 1024)

        total_gc_ms = young_gc_total_ms + full_gc_total_ms
        # Estimate duration from collection (default 60s)
        duration_sec = data.get("jfr_duration_sec", 60)
        gc_pause_ratio = total_gc_ms / (duration_sec * 1000) if duration_sec > 0 else 0.0
        gc_pause_ratio = min(gc_pause_ratio, 1.0)

        total_gc_count = young_gc_count + full_gc_count
        avg_gc_pause_ms = total_gc_ms / total_gc_count if total_gc_count > 0 else 0.0

        heap_usage_ratio = heap_used_mb / heap_max_mb if heap_max_mb > 0 else 0.0

        return GCMetrics(
            gc_algorithm=gc_algorithm,
            young_gc_count=young_gc_count,
            young_gc_total_ms=round(young_gc_total_ms, 1),
            full_gc_count=full_gc_count,
            full_gc_total_ms=round(full_gc_total_ms, 1),
            gc_pause_ratio=round(gc_pause_ratio, 4),
            avg_gc_pause_ms=round(avg_gc_pause_ms, 1),
            max_gc_pause_ms=round(full_gc_total_ms, 1) if full_gc_count > 0 else round(young_gc_total_ms / max(young_gc_count, 1), 1),
            heap_used_mb=round(heap_used_mb, 1),
            heap_max_mb=round(heap_max_mb, 1),
            heap_usage_ratio=round(heap_usage_ratio, 4),
            metaspace_used_mb=round(metaspace_used_mb, 1),
        )

    def _extract_jit_from_jfr(self, data: Dict[str, Any]) -> JITMetrics:
        """Extract JIT metrics from JFR parsed data."""
        jfr_parsed = data.get("jfr_parsed", {})
        jit_events = jfr_parsed.get("jit_events", [])

        total_compilations = 0
        c1_count = 0
        c2_count = 0
        osr_count = 0
        deoptimization_count = 0

        for event in jit_events:
            event_type = event.get("type", "")
            props = event.get("values", event)

            if "Compilation" in event_type:
                total_compilations += 1
                compiler = str(props.get("compiler", "")).lower()
                if "c1" in compiler:
                    c1_count += 1
                elif "c2" in compiler:
                    c2_count += 1
                if props.get("method", {}).get("type", "") == "OSR":
                    osr_count += 1
            elif "Deoptimization" in event_type:
                deoptimization_count += 1

        duration_sec = data.get("jfr_duration_sec", 60)
        compilations_per_sec = total_compilations / duration_sec if duration_sec > 0 else 0.0
        deopt_ratio = deoptimization_count / total_compilations if total_compilations > 0 else 0.0

        return JITMetrics(
            total_compilations=total_compilations,
            compilations_per_sec=round(compilations_per_sec, 2),
            deoptimization_count=deoptimization_count,
            deopt_ratio=round(deopt_ratio, 4),
            c1_count=c1_count,
            c2_count=c2_count,
            osr_count=osr_count,
        )

    def _extract_threads_from_jfr(self, data: Dict[str, Any]) -> JVMThreadMetrics:
        """Extract thread and safepoint metrics from JFR parsed data."""
        jfr_parsed = data.get("jfr_parsed", {})
        thread_events = jfr_parsed.get("thread_events", [])
        safepoint_events = jfr_parsed.get("safepoint_events", [])

        total_threads = 0
        daemon_threads = 0
        active_threads = 0
        has_thread_stats = False
        for event in thread_events:
            event_type = event.get("type", "")
            if "ThreadStatistics" in event_type:
                props = event.get("values", event)
                total_threads = props.get("activeCount", total_threads)
                daemon_threads = props.get("daemonCount", daemon_threads)
                active_threads = props.get("runningCount", 0)
                has_thread_stats = True

        # Fallback: count ThreadStart events if ThreadStatistics not available
        if not has_thread_stats:
            for event in thread_events:
                event_type = event.get("type", "")
                if "ThreadStart" in event_type:
                    total_threads += 1
                    props = event.get("values", event)
                    if props.get("daemon", False):
                        daemon_threads += 1

        # Fallback: count ThreadStart events if ThreadStatistics not available
        if total_threads == 0:
            for event in thread_events:
                event_type = event.get("type", "")
                if "ThreadStart" in event_type:
                    total_threads += 1

        safepoint_count = len(safepoint_events) // 2  # begin+end pairs
        safepoint_total_ms = 0.0
        for event in safepoint_events:
            props = event.get("values", event)
            duration_ns = props.get("duration", 0)
            if isinstance(duration_ns, (int, float)):
                safepoint_total_ms += duration_ns / 1_000_000

        duration_sec = data.get("jfr_duration_sec", 60)
        safepoint_ratio = safepoint_total_ms / (duration_sec * 1000) if duration_sec > 0 else 0.0

        return JVMThreadMetrics(
            total_threads=total_threads,
            active_threads=active_threads,
            daemon_threads=daemon_threads,
            safepoint_count=safepoint_count,
            safepoint_total_ms=round(safepoint_total_ms, 1),
            safepoint_ratio=round(min(safepoint_ratio, 1.0), 4),
        )

    def _extract_gc_from_jstat(self, data: Dict[str, Any]) -> GCMetrics:
        """Extract GC metrics from jstat output (JDK 8 fallback)."""
        jstat = data.get("jstat_parsed", {})
        gcutil = data.get("gcutil_parsed", {})

        # jstat -gc columns: EC/EU/OC/OU/MC/MU...
        # Estimate from gcutil: FGC (full GC count), YGC (young GC count)
        young_gc_count = int(jstat.get("YGC", gcutil.get("YGC", 0)))
        full_gc_count = int(jstat.get("FGC", gcutil.get("FGC", 0)))

        young_gc_total_ms = float(jstat.get("YGCT", 0))
        full_gc_total_ms = float(jstat.get("FGCT", 0))

        # Heap usage from gcutil (percentage)
        heap_usage_pct = float(gcutil.get("O", 0)) / 100.0 if gcutil.get("O") else 0.0

        # Estimate heap sizes from jstat -gc (KB values)
        old_used_kb = float(jstat.get("OU", 0))
        old_max_kb = float(jstat.get("OC", 0))
        eden_used_kb = float(jstat.get("EU", 0))
        eden_max_kb = float(jstat.get("EC", 0))

        heap_used_mb = (old_used_kb + eden_used_kb) / 1024
        heap_max_mb = (old_max_kb + eden_max_kb) / 1024

        total_gc_ms = young_gc_total_ms + full_gc_total_ms
        # Approximate duration from sample count
        sample_count = jstat.get("_sample_count", 60)
        duration_sec = max(sample_count, 1)
        gc_pause_ratio = total_gc_ms / (duration_sec * 1000) if duration_sec > 0 else 0.0

        total_gc_count = young_gc_count + full_gc_count
        avg_gc_pause_ms = total_gc_ms / total_gc_count if total_gc_count > 0 else 0.0

        metaspace_mb = float(jstat.get("MU", 0)) / 1024

        return GCMetrics(
            gc_algorithm="unknown",
            young_gc_count=young_gc_count,
            young_gc_total_ms=round(young_gc_total_ms, 1),
            full_gc_count=full_gc_count,
            full_gc_total_ms=round(full_gc_total_ms, 1),
            gc_pause_ratio=round(min(gc_pause_ratio, 1.0), 4),
            avg_gc_pause_ms=round(avg_gc_pause_ms, 1),
            max_gc_pause_ms=round(full_gc_total_ms, 1) if full_gc_count > 0 else 0.0,
            heap_used_mb=round(heap_used_mb, 1),
            heap_max_mb=round(heap_max_mb, 1),
            heap_usage_ratio=round(heap_usage_pct, 4),
            metaspace_used_mb=round(metaspace_mb, 1),
        )

    def _extract_threads_from_jstat(self, data: Dict[str, Any]) -> JVMThreadMetrics:
        """Extract thread metrics from jstack output (JDK 8 fallback)."""
        jstack = data.get("jstack_parsed", {})
        return JVMThreadMetrics(
            total_threads=jstack.get("total_threads", 0),
            daemon_threads=jstack.get("daemon_threads", 0),
            deadlocked_threads=jstack.get("deadlocked_threads", 0),
        )
