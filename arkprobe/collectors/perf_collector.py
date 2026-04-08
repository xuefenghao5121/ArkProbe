"""Linux perf wrapper for ARM PMU event collection on Kunpeng processors.

Supports:
- perf stat with JSON output and multiplexing correction
- perf record for sampling-based profiling
- ARM TopDown Level 1 analysis
- Sequential event group collection to avoid multiplexing
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .arm_events import (
    CORE_EVENT_GROUPS,
    UNCORE_EVENT_GROUPS,
    EventGroup,
    build_perf_event_string,
    get_all_core_event_groups,
    get_kunpeng_model,
)
from .base import BaseCollector, CollectionResult
from ..utils.process import RunResult, run_cmd

log = logging.getLogger(__name__)


@dataclass
class PerfCounter:
    """A single perf counter value with multiplexing correction."""
    event: str
    value: float
    unit: str = ""
    enabled_ns: float = 0
    running_ns: float = 0

    @property
    def multiplexing_ratio(self) -> float:
        """Fraction of time this counter was actually counting."""
        if self.enabled_ns <= 0:
            return 0.0
        return self.running_ns / self.enabled_ns

    @property
    def corrected_value(self) -> float:
        """Value corrected for multiplexing."""
        ratio = self.multiplexing_ratio
        if ratio <= 0:
            return 0.0
        return self.value / ratio


@dataclass
class PerfStatResult:
    """Parsed result from a perf stat run."""
    group_name: str
    counters: Dict[str, PerfCounter] = field(default_factory=dict)
    duration_sec: float = 0.0
    raw_output: str = ""

    def get(self, name: str, corrected: bool = True) -> float:
        """Get counter value by logical name."""
        if name not in self.counters:
            return 0.0
        return self.counters[name].corrected_value if corrected else self.counters[name].value


class PerfCollector(BaseCollector):
    """Wraps Linux perf for ARM PMU event collection."""

    def __init__(self, output_dir: Path, kunpeng_model: str = "920"):
        super().__init__(output_dir)
        self.model = get_kunpeng_model(kunpeng_model)
        self._available_events: Optional[set] = None

    def collect(self, command: str = "", pid: Optional[int] = None,
                duration_sec: int = 60, **kwargs) -> CollectionResult:
        """Run full perf stat collection across all event groups."""
        results = self.stat_all_groups(
            command=command, pid=pid, duration_sec=duration_sec
        )
        data = {}
        errors = []
        raw_files = {}
        for group_name, stat_result in results.items():
            data[group_name] = {
                name: counter.corrected_value
                for name, counter in stat_result.counters.items()
            }
            data[f"{group_name}_duration"] = stat_result.duration_sec
            raw_path = self._save_raw(f"perf_stat_{group_name}.json", stat_result.raw_output)
            raw_files[group_name] = raw_path

        return CollectionResult(
            collector_name="perf",
            data=data,
            raw_files=raw_files,
            errors=errors,
        )

    def stat(
        self,
        event_group: EventGroup,
        command: str = "",
        pid: Optional[int] = None,
        duration_sec: Optional[int] = None,
        cpu_list: Optional[str] = None,
        repeat: int = 3,
    ) -> PerfStatResult:
        """Run perf stat for a single event group.

        Uses JSON output (-j) and applies multiplexing correction.
        """
        event_str = build_perf_event_string(event_group, self.model.pmu_name)
        cmd = ["perf", "stat", "-j", "-e", event_str, "-r", str(repeat)]

        if cpu_list:
            cmd.extend(["-C", cpu_list])

        if pid is not None:
            cmd.extend(["-p", str(pid)])
            if duration_sec:
                cmd.extend(["--", "sleep", str(duration_sec)])
        elif command:
            cmd.extend(["--", "sh", "-c", command])
        elif duration_sec:
            cmd.extend(["-a", "--", "sleep", str(duration_sec)])
        else:
            cmd.extend(["-a", "--", "sleep", "10"])

        result = run_cmd(cmd, timeout_sec=max((duration_sec or 60) * repeat + 60, 300))

        # perf stat outputs JSON to stderr
        raw_output = result.stderr
        counters = self._parse_perf_stat_json(raw_output, event_group)

        # Extract duration from perf stat output
        duration = self._extract_duration(raw_output, duration_sec or 10)

        return PerfStatResult(
            group_name=event_group.name,
            counters=counters,
            duration_sec=duration,
            raw_output=raw_output,
        )

    def stat_all_groups(
        self,
        command: str = "",
        pid: Optional[int] = None,
        duration_sec: int = 60,
        repeat: int = 3,
    ) -> Dict[str, PerfStatResult]:
        """Run perf stat for ALL core event groups sequentially.

        Sequential collection avoids excessive multiplexing and ensures
        each group gets dedicated use of the 6 programmable PMU counters.
        """
        results = {}
        for group in get_all_core_event_groups():
            self.log.info("Collecting event group: %s", group.name)
            try:
                stat_result = self.stat(
                    event_group=group,
                    command=command,
                    pid=pid,
                    duration_sec=duration_sec,
                    repeat=repeat,
                )
                results[group.name] = stat_result
            except Exception as e:
                self.log.error("Failed to collect %s: %s", group.name, e)
        return results

    def record(
        self,
        command: str = "",
        events: Optional[List[str]] = None,
        frequency: int = 99,
        call_graph: str = "fp",
        duration_sec: Optional[int] = None,
        pid: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> Path:
        """Run perf record for sampling-based profiling."""
        if output_file is None:
            output_file = str(self.output_dir / "perf.data")

        cmd = ["perf", "record", "-o", output_file, "-F", str(frequency),
               "-g", f"--call-graph={call_graph}"]

        if events:
            cmd.extend(["-e", ",".join(events)])

        if pid is not None:
            cmd.extend(["-p", str(pid)])
            if duration_sec:
                cmd.extend(["--", "sleep", str(duration_sec)])
        elif command:
            cmd.extend(["--", "sh", "-c", command])
        elif duration_sec:
            cmd.extend(["-a", "--", "sleep", str(duration_sec)])

        result = run_cmd(cmd, timeout_sec=max((duration_sec or 60) + 60, 300))
        if not result.ok:
            self.log.error("perf record failed: %s", result.stderr[:500])

        return Path(output_file)

    def topdown_analysis(
        self,
        command: str = "",
        pid: Optional[int] = None,
        duration_sec: int = 30,
    ) -> Dict[str, float]:
        """Run TopDown Level 1 analysis using ARM stall events.

        ARM TopDown L1:
          frontend_bound = stall_frontend / cycles
          backend_bound  = stall_backend / cycles
          retiring       = (instructions / cycles) / dispatch_width
          bad_speculation = 1.0 - frontend - backend - retiring  (clamped)
        """
        stat_result = self.stat(
            CORE_EVENT_GROUPS["topdown_l1"],
            command=command,
            pid=pid,
            duration_sec=duration_sec,
        )

        cycles = stat_result.get("cycles")
        instructions = stat_result.get("instructions")
        stall_fe = stat_result.get("stall_frontend")
        stall_be = stat_result.get("stall_backend")

        if cycles <= 0:
            return {"frontend_bound": 0, "backend_bound": 0,
                    "retiring": 0, "bad_speculation": 0}

        frontend_bound = stall_fe / cycles
        backend_bound = stall_be / cycles
        retiring = (instructions / cycles) / self.model.dispatch_width
        bad_speculation = 1.0 - frontend_bound - backend_bound - retiring

        # Clamp bad_speculation (can go negative due to measurement noise)
        bad_speculation = max(0.0, min(1.0, bad_speculation))

        # Rebalance if total exceeds 1.0
        total = frontend_bound + backend_bound + retiring + bad_speculation
        if total > 0:
            frontend_bound /= total
            backend_bound /= total
            retiring /= total
            bad_speculation /= total

        return {
            "frontend_bound": round(frontend_bound, 4),
            "backend_bound": round(backend_bound, 4),
            "retiring": round(retiring, 4),
            "bad_speculation": round(bad_speculation, 4),
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _parse_perf_stat_json(
        self, output: str, group: EventGroup
    ) -> Dict[str, PerfCounter]:
        """Parse perf stat JSON output (one JSON object per line)."""
        counters: Dict[str, PerfCounter] = {}
        event_to_name = {v: k for k, v in group.events.items()}

        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_raw = obj.get("event", "")
            counter_value = obj.get("counter-value", "0")
            unit = obj.get("unit", "")

            # Clean up event name to match our definitions
            event_clean = event_raw.strip()
            # Remove PMU prefix: "armv8_pmuv3/cpu_cycles/" -> "cpu_cycles"
            m = re.search(r"/([^/]+)/", event_clean)
            if m:
                event_clean = m.group(1)

            # Map to logical name
            logical_name = event_to_name.get(event_clean, event_clean)

            try:
                value = float(str(counter_value).replace(",", ""))
            except ValueError:
                value = 0.0

            counters[logical_name] = PerfCounter(
                event=event_raw,
                value=value,
                unit=unit,
                enabled_ns=float(obj.get("event-runtime", 0)),
                running_ns=float(obj.get("pcnt-running", 100)) / 100.0
                           * float(obj.get("event-runtime", 0)),
            )

        return counters

    def _extract_duration(self, raw_output: str, default: float) -> float:
        """Extract elapsed time from perf stat output."""
        m = re.search(r"([\d.]+)\s+seconds time elapsed", raw_output)
        if m:
            return float(m.group(1))
        return default

    def detect_available_events(self) -> set:
        """Probe which PMU events are available on current hardware."""
        if self._available_events is not None:
            return self._available_events

        self._available_events = set()
        for group in get_all_core_event_groups():
            for name, event in group.events.items():
                event_str = f"{self.model.pmu_name}/{event}/"
                result = run_cmd(
                    ["perf", "stat", "-e", event_str, "true"],
                    timeout_sec=10,
                )
                if result.ok and "not supported" not in result.stderr:
                    self._available_events.add(name)

        self.log.info("Detected %d available PMU events", len(self._available_events))
        return self._available_events
