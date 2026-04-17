"""Collector orchestrator - coordinates all collectors for a scenario run.

Manages the full collection pipeline:
1. Pre-collection: system topology, platform info
2. Warm-up: run workload briefly without collection
3. Phase 2+3+4 (parallel): perf stat + eBPF + Uncore PMU
4. System metrics collection
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import CollectionResult
from .ebpf_collector import EbpfCollector
from .perf_collector import PerfCollector, validate_command_safety
from .system_collector import SystemCollector

log = logging.getLogger(__name__)


@dataclass
class ScenarioCollectionConfig:
    """Configuration for a single scenario collection run."""
    scenario_name: str
    workload_command: str
    target_pid: Optional[int] = None
    kunpeng_model: str = "920"
    perf_duration_sec: int = 60
    ebpf_duration_sec: int = 30
    warmup_sec: int = 30
    ebpf_probes: List[str] = field(default_factory=lambda: [
        "io_latency", "lock_contention", "cache_stats", "tcp_latency",
    ])
    skip_ebpf: bool = False
    skip_scalability: bool = True
    scalability_core_counts: List[int] = field(default_factory=lambda: [
        1, 2, 4, 8, 16, 32, 48, 64,
    ])
    cache_ttl_sec: int = 0
    force: bool = False


@dataclass
class FullCollectionResult:
    """Result from a complete scenario collection run."""
    scenario_name: str
    perf_data: Dict[str, Any] = field(default_factory=dict)
    ebpf_data: Dict[str, Any] = field(default_factory=dict)
    system_data: Dict[str, Any] = field(default_factory=dict)
    scalability_data: Optional[Dict[str, Any]] = None
    raw_files: Dict[str, Path] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    collection_duration_sec: float = 0.0

    def save(self, output_dir: Path) -> Path:
        """Save all collected data to a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"{self.scenario_name}_raw.json"
        data = {
            "scenario_name": self.scenario_name,
            "perf": self.perf_data,
            "ebpf": self.ebpf_data,
            "system": self.system_data,
            "scalability": self.scalability_data,
            "errors": self.errors,
            "collection_duration_sec": self.collection_duration_sec,
        }
        result_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return result_file

    @classmethod
    def load(cls, path: Path) -> "FullCollectionResult":
        """Load from a saved JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            scenario_name=data["scenario_name"],
            perf_data=data.get("perf", {}),
            ebpf_data=data.get("ebpf", {}),
            system_data=data.get("system", {}),
            scalability_data=data.get("scalability"),
            errors=data.get("errors", []),
            collection_duration_sec=data.get("collection_duration_sec", 0),
        )


class CollectorOrchestrator:
    """Coordinates all collectors for a single scenario run."""

    def __init__(self, config: ScenarioCollectionConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir / config.scenario_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.perf = PerfCollector(
            output_dir=self.output_dir / "perf",
            kunpeng_model=config.kunpeng_model,
        )
        self.ebpf = EbpfCollector(output_dir=self.output_dir / "ebpf")
        self.system = SystemCollector(output_dir=self.output_dir / "system")

    def _find_cached_result(self) -> Optional[FullCollectionResult]:
        """Check for a recent cached result file within TTL."""
        if self.config.cache_ttl_sec <= 0 or self.config.force:
            return None

        result_file = self.output_dir / f"{self.config.scenario_name}_raw.json"
        if not result_file.exists():
            return None

        try:
            mtime = result_file.stat().st_mtime
            age = time.time() - mtime
            if age <= self.config.cache_ttl_sec:
                log.info("[%s] Using cached result (age=%.0fs, ttl=%ds)",
                         self.config.scenario_name, age, self.config.cache_ttl_sec)
                return FullCollectionResult.load(result_file)
        except OSError:
            pass

        return None

    def run(self) -> FullCollectionResult:
        """Execute the complete collection pipeline.

        Phases 2 (perf), 3 (eBPF), and 4 (Uncore) run in parallel
        since they can observe the same workload concurrently.
        System info is collected first (fast, provides platform context).
        """
        start_time = time.time()
        result = FullCollectionResult(scenario_name=self.config.scenario_name)

        # Check cache first
        cached = self._find_cached_result()
        if cached is not None:
            return cached

        # Phase 0: System info (always first, fast)
        log.info("[%s] Phase 0: Collecting system info", self.config.scenario_name)
        try:
            sys_result = self.system.collect(duration_sec=5)
            result.system_data = sys_result.data
            result.raw_files.update(sys_result.raw_files)
            result.errors.extend(sys_result.errors)
        except Exception as e:
            log.error("System collection failed: %s", e)
            result.errors.append(f"system: {e}")

        # Phase 1: Warmup (if configured)
        if self.config.warmup_sec > 0 and self.config.workload_command:
            log.info("[%s] Phase 1: Warming up for %ds",
                     self.config.scenario_name, self.config.warmup_sec)
            time.sleep(self.config.warmup_sec)

        # Phases 2+3+4: perf stat, eBPF, Uncore in parallel
        def run_perf():
            log.info("[%s] Phase 2: perf stat collection (%ds per group)",
                     self.config.scenario_name, self.config.perf_duration_sec)
            try:
                return self.perf.collect(
                    command=self.config.workload_command,
                    pid=self.config.target_pid,
                    duration_sec=self.config.perf_duration_sec,
                )
            except Exception as e:
                log.error("Perf collection failed: %s", e)
                return CollectionResult(collector_name="perf", errors=[f"perf: {e}"])

        def run_ebpf():
            if self.config.skip_ebpf:
                return None
            log.info("[%s] Phase 3: eBPF tracing (%ds)",
                     self.config.scenario_name, self.config.ebpf_duration_sec)
            try:
                return self.ebpf.collect(
                    pid=self.config.target_pid,
                    duration_sec=self.config.ebpf_duration_sec,
                    probes=self.config.ebpf_probes,
                )
            except Exception as e:
                log.error("eBPF collection failed: %s", e)
                return CollectionResult(collector_name="ebpf", errors=[f"ebpf: {e}"])

        def run_uncore():
            log.info("[%s] Phase 4: Uncore PMU collection (%ds)",
                     self.config.scenario_name, self.config.perf_duration_sec)
            try:
                return self.perf.collect_uncore(
                    duration_sec=self.config.perf_duration_sec,
                )
            except Exception as e:
                log.warning("Uncore collection failed (non-critical): %s", e)
                return None

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(run_perf): "perf",
                pool.submit(run_ebpf): "ebpf",
                pool.submit(run_uncore): "uncore",
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    col_result = future.result()
                except Exception as e:
                    result.errors.append(f"{label}: {e}")
                    continue

                if label == "perf" and col_result is not None:
                    result.perf_data = col_result.data
                    result.raw_files.update(col_result.raw_files)
                    result.errors.extend(col_result.errors)
                elif label == "ebpf" and col_result is not None:
                    result.ebpf_data = col_result.data
                    result.raw_files.update(col_result.raw_files)
                    result.errors.extend(col_result.errors)
                elif label == "uncore" and col_result is not None:
                    result.perf_data["uncore"] = col_result

        result.collection_duration_sec = time.time() - start_time
        log.info("[%s] Collection complete in %.1fs with %d errors",
                 self.config.scenario_name, result.collection_duration_sec,
                 len(result.errors))

        # Save consolidated results
        result.save(self.output_dir)
        return result

    def run_scalability_sweep(
        self,
        workload_factory: Callable[[int], str],
        throughput_parser: Callable[[str], float],
    ) -> Dict[str, Any]:
        """Run abbreviated perf stat at different core counts.

        Args:
            workload_factory: Given core count, returns the command string
            throughput_parser: Extracts throughput metric from workload stdout
        """
        core_counts = self.config.scalability_core_counts
        throughputs = []
        perf_summaries = []

        for cores in core_counts:
            log.info("[%s] Scalability sweep: %d cores",
                     self.config.scenario_name, cores)

            command = workload_factory(cores)
            cpu_list = f"0-{cores - 1}"

            # Short perf stat for TopDown at this core count
            from .arm_events import CORE_EVENT_GROUPS
            topdown = self.perf.stat(
                CORE_EVENT_GROUPS["topdown_l1"],
                command=command,
                cpu_list=cpu_list,
                duration_sec=30,
                repeat=1,
            )

            # Get throughput from workload output
            from ..utils.process import run_cmd
            # Security: validate command before shell execution
            is_safe, error_msg = validate_command_safety(command)
            if not is_safe:
                raise ValueError(f"Unsafe command rejected: {error_msg}")
            wl_result = run_cmd(
                ["sh", "-c", command],
                timeout_sec=120,
            )
            tp = throughput_parser(wl_result.stdout)
            throughputs.append(tp)

            perf_summaries.append({
                "core_count": cores,
                "ipc": topdown.get("instructions", 0) / max(topdown.get("cycles", 1), 1),
                "frontend_bound": topdown.get("stall_frontend", 0) / max(topdown.get("cycles", 1), 1),
                "backend_bound": topdown.get("stall_backend", 0) / max(topdown.get("cycles", 1), 1),
            })

        # Compute scaling efficiency
        base_tp = throughputs[0] if throughputs else 1.0
        efficiencies = []
        for i, tp in enumerate(throughputs):
            ideal = base_tp * core_counts[i]
            efficiencies.append(tp / ideal if ideal > 0 else 0.0)

        return {
            "core_counts": core_counts,
            "throughputs": throughputs,
            "efficiencies": efficiencies,
            "perf_summaries": perf_summaries,
        }
