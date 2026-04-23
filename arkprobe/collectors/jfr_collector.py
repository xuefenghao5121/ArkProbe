"""JFR (Java Flight Recorder) collector for JVM introspection.

Supports:
- JDK 11+: JFR via jcmd (built-in, no extra flags needed)
- JDK 8: fallback to jstat -gc sampling + GC log parsing
- Configurable event profiles (gc, jit, thread, memory)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseCollector, CollectionResult
from ..utils.process import RunResult, run_cmd

log = logging.getLogger(__name__)

JFR_EVENT_PROFILES: Dict[str, List[str]] = {
    "gc": [
        "jdk.GCPhaseLevel",
        "jdk.GCHeapSummary",
        "jdk.GCConfiguration",
        "jdk.YoungGCPhase",
        "jdk.OldGCPhase",
    ],
    "jit": [
        "jdk.Compilation",
        "jdk.CompilerInlining",
        "jdk.Deoptimization",
    ],
    "thread": [
        "jdk.ThreadStart",
        "jdk.ThreadEnd",
        "jdk.JavaMonitorWait",
        "jdk.JavaMonitorInflate",
        "jdk.SafepointBegin",
        "jdk.SafepointEnd",
        "jdk.ThreadStatistics",
    ],
    "memory": [
        "jdk.GCHeapSummary",
        "jdk.MetaspaceSummary",
        "jdk.PromotedObject",
    ],
}

DEFAULT_JFR_PROFILES = ["gc", "jit", "thread"]


def resolve_jfr_events(profiles: List[str]) -> str:
    """Resolve event profile names to a JFR event string."""
    events: List[str] = []
    for p in profiles:
        if p in JFR_EVENT_PROFILES:
            events.extend(JFR_EVENT_PROFILES[p])
        else:
            events.append(p)
    return ",".join(events)


def detect_jdk_version(pid: int) -> str:
    """Detect JDK version of a running JVM process via jcmd."""
    result = run_cmd(["jcmd", str(pid), "VM.version"], timeout_sec=10)
    if not result.ok:
        return "unknown"
    for line in result.stdout.splitlines():
        if "version" in line.lower():
            return line.strip()
    return result.stdout.strip()[:120]


def _major_jdk_version(jdk_version: str) -> int:
    """Extract major JDK version number."""
    # JDK 8 reports as "1.8.0_xxx"
    m = re.search(r"1\.8", jdk_version)
    if m:
        return 8
    m = re.search(r"(\d+)", jdk_version)
    if not m:
        return 0
    return int(m.group(1))


class JfrCollector(BaseCollector):
    """JFR (Java Flight Recorder) collector for JVM introspection."""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir=output_dir)

    def collect(self, **kwargs) -> CollectionResult:
        """Collect JVM runtime data via JFR or jstat fallback.

        kwargs:
            target_pid: int          JVM process PID
            jfr_duration_sec: int    Recording duration (default 60)
            jfr_events: list[str]    Event profile names (default ["gc","jit","thread"])
        """
        pid = kwargs.get("target_pid")
        if pid is None:
            return CollectionResult(
                collector_name="jfr",
                errors=["No target_pid provided for JFR collection"],
            )

        duration = kwargs.get("jfr_duration_sec", 60)
        event_profiles = kwargs.get("jfr_events", DEFAULT_JFR_PROFILES)

        jdk_version = detect_jdk_version(pid)
        major = _major_jdk_version(jdk_version)

        if major >= 11:
            result = self._collect_jfr(pid, duration, event_profiles, jdk_version)
        else:
            result = self._collect_jdk8_fallback(pid, duration, jdk_version)

        return result

    def _collect_jfr(
        self,
        pid: int,
        duration_sec: int,
        event_profiles: List[str],
        jdk_version: str,
    ) -> CollectionResult:
        """JDK 11+ path: start JFR recording, wait, parse JSON output."""
        errors: list[str] = []
        data: Dict[str, Any] = {
            "jdk_version": jdk_version,
            "jfr_available": True,
            "jfr_duration_sec": duration_sec,
        }
        raw_files: Dict[str, Path] = {}

        jfr_file = self.output_dir / f"arkprobe-{pid}.jfr"
        recording_name = "arkprobe"

        # Start recording
        events_str = resolve_jfr_events(event_profiles)
        start_cmd = [
            "jcmd", str(pid), "JFR.start",
            f"name={recording_name}",
            f"settings={events_str}",
            f"duration={duration_sec}s",
            f"filename={jfr_file}",
        ]
        log.info("Starting JFR recording: %s", " ".join(start_cmd))
        start_result = run_cmd(start_cmd, timeout_sec=30)
        if not start_result.ok:
            # JFR may already be running; try to stop old recording first
            run_cmd(["jcmd", str(pid), "JFR.stop", f"name={recording_name}"],
                    timeout_sec=10)
            start_result = run_cmd(start_cmd, timeout_sec=30)
            if not start_result.ok:
                return CollectionResult(
                    collector_name="jfr",
                    errors=[f"JFR.start failed: {start_result.stderr[:300]}"],
                )

        # Wait for recording to finish
        log.info("JFR recording in progress (%ds)...", duration_sec)
        time.sleep(duration_sec + 2)

        # Verify file exists
        if not jfr_file.exists():
            errors.append(f"JFR output file not found: {jfr_file}")
            return CollectionResult(collector_name="jfr", data=data,
                                    errors=errors)

        raw_files["jfr_binary"] = jfr_file

        # Parse via jfr print --json
        json_file = self.output_dir / f"arkprobe-{pid}.json"
        print_result = run_cmd(
            ["jfr", "print", "--json", str(jfr_file)],
            timeout_sec=120,
        )
        if print_result.ok and print_result.stdout:
            json_file.write_text(print_result.stdout, encoding="utf-8")
            raw_files["jfr_json"] = json_file
            data["jfr_parsed"] = self._parse_jfr_json(print_result.stdout)
        else:
            errors.append(f"jfr print failed: {print_result.stderr[:300]}")

        data["jfr_events_collected"] = event_profiles

        return CollectionResult(
            collector_name="jfr",
            data=data,
            raw_files=raw_files,
            errors=errors,
        )

    def _collect_jdk8_fallback(
        self,
        pid: int,
        duration_sec: int,
        jdk_version: str,
    ) -> CollectionResult:
        """JDK 8 fallback: jstat -gc sampling + jstack thread snapshot."""
        errors: list[str] = []
        data: Dict[str, Any] = {
            "jdk_version": jdk_version,
            "jfr_available": False,
            "jfr_duration_sec": duration_sec,
        }
        raw_files: Dict[str, Path] = {}

        # jstat -gc sampling at 1s intervals
        interval_ms = 1000
        sample_count = duration_sec
        jstat_result = run_cmd(
            ["jstat", "-gc", str(pid), str(interval_ms), str(sample_count)],
            timeout_sec=duration_sec + 30,
        )
        if jstat_result.ok:
            raw_path = self._save_raw(f"jstat-gc-{pid}.txt", jstat_result.stdout)
            raw_files["jstat_gc"] = raw_path
            data["jstat_parsed"] = self._parse_jstat_gc(jstat_result.stdout)
        else:
            errors.append(f"jstat -gc failed: {jstat_result.stderr[:300]}")

        # jstack thread dump (single snapshot)
        jstack_result = run_cmd(["jstack", str(pid)], timeout_sec=15)
        if jstack_result.ok:
            raw_path = self._save_raw(f"jstack-{pid}.txt", jstack_result.stdout)
            raw_files["jstack"] = raw_path
            data["jstack_parsed"] = self._parse_jstack(jstack_result.stdout)
        else:
            errors.append(f"jstack failed: {jstack_result.stderr[:300]}")

        # jstat -gcutil for summary
        gcutil_result = run_cmd(
            ["jstat", "-gcutil", str(pid), str(interval_ms), str(sample_count)],
            timeout_sec=duration_sec + 30,
        )
        if gcutil_result.ok:
            raw_path = self._save_raw(f"jstat-gcutil-{pid}.txt", gcutil_result.stdout)
            raw_files["jstat_gcutil"] = raw_path
            data["gcutil_parsed"] = self._parse_jstat_gcutil(gcutil_result.stdout)
        else:
            errors.append(f"jstat -gcutil failed: {gcutil_result.stderr[:300]}")

        return CollectionResult(
            collector_name="jfr",
            data=data,
            raw_files=raw_files,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # JFR JSON parsing
    # ------------------------------------------------------------------

    def _parse_jfr_json(self, json_str: str) -> Dict[str, Any]:
        """Extract key metrics from JFR JSON output."""
        parsed: Dict[str, Any] = {
            "gc_events": [],
            "jit_events": [],
            "safepoint_events": [],
            "thread_events": [],
        }
        try:
            jfr_data = json.loads(json_str)
        except json.JSONDecodeError:
            return parsed

        events = jfr_data.get("events", [])
        for event in events:
            event_type = event.get("type", "")
            if "GC" in event_type or "gc" in event_type.lower():
                parsed["gc_events"].append(event)
            elif "Compilation" in event_type or "Deoptimization" in event_type:
                parsed["jit_events"].append(event)
            elif "Safepoint" in event_type:
                parsed["safepoint_events"].append(event)
            elif "Thread" in event_type:
                parsed["thread_events"].append(event)

        return parsed

    # ------------------------------------------------------------------
    # jstat parsing (JDK 8 fallback)
    # ------------------------------------------------------------------

    def _parse_jstat_gc(self, output: str) -> Dict[str, Any]:
        """Parse jstat -gc tabular output into structured data."""
        lines = output.strip().splitlines()
        if len(lines) < 2:
            return {}

        headers = lines[0].split()
        # Take last data row as the final sample
        last_row = lines[-1].split()
        result: Dict[str, Any] = {}
        for i, h in enumerate(headers):
            if i < len(last_row):
                try:
                    result[h] = float(last_row[i])
                except ValueError:
                    result[h] = last_row[i]

        # Calculate deltas if multiple samples exist
        if len(lines) > 2:
            first_row = lines[1].split()
            result["_sample_count"] = len(lines) - 1
            result["_first_sample"] = {}
            for i, h in enumerate(headers):
                if i < len(first_row):
                    try:
                        result["_first_sample"][h] = float(first_row[i])
                    except ValueError:
                        result["_first_sample"][h] = first_row[i]

        return result

    def _parse_jstat_gcutil(self, output: str) -> Dict[str, Any]:
        """Parse jstat -gcutil output (percentage-based)."""
        lines = output.strip().splitlines()
        if len(lines) < 2:
            return {}

        headers = lines[0].split()
        last_row = lines[-1].split()
        result: Dict[str, Any] = {}
        for i, h in enumerate(headers):
            if i < len(last_row):
                try:
                    result[h] = float(last_row[i])
                except ValueError:
                    result[h] = last_row[i]
        return result

    def _parse_jstack(self, output: str) -> Dict[str, Any]:
        """Parse jstack output for thread counts and deadlock info."""
        result: Dict[str, Any] = {
            "total_threads": 0,
            "daemon_threads": 0,
            "deadlocked_threads": 0,
            "blocked_threads": 0,
        }

        for line in output.splitlines():
            if "lang.Thread" in line or "daemon" in line.lower():
                result["total_threads"] += 1
            if "daemon" in line.lower():
                result["daemon_threads"] += 1
            if "BLOCKED" in line:
                result["blocked_threads"] += 1
            if "Found one Java-level deadlock" in line:
                result["deadlocked_threads"] = -1  # flag that deadlock exists

        return result
