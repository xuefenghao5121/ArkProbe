"""
JFR-based hotspot profiler for identifying CPU-intensive Java methods.

Extracts hotspot methods from JFR ExecutionSample and ProfiledMethod events.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from arkprobe.collectors.jfr_collector import JfrCollector
from arkprobe.utils.process import run_cmd
from arkprobe.hotspot.models import HotspotMethod, HotspotProfile

log = logging.getLogger(__name__)

# Extended JFR event profiles for hotspot detection
JFR_HOTSPOT_PROFILES = {
    "profiles": [
        "jdk.ExecutionSample",
        "jdk.ProfiledMethod",
        "jdk.Compilation",
        "jdk.CompilerInlining",
        "jdk.Deoptimization",
    ],
}


def _detect_jdk_version(pid: int) -> str:
    """Detect JDK version of a running JVM process."""
    result = run_cmd(["jcmd", str(pid), "VM.version"], timeout_sec=10)
    if result.ok:
        for line in result.stdout.splitlines():
            if "version" in line.lower():
                return line.strip()
    return "unknown"


def _parse_stack_trace(frame: dict) -> tuple[str, str, str]:
    """Parse a stack frame to extract class, method, and signature."""
    method = frame.get("method", {})
    klass = method.get("type", {})

    class_name = klass.get("name", "Unknown")
    method_name = method.get("name", "unknown")
    signature = method.get("descriptor", "")

    return class_name, method_name, signature


def _is_hotspot_candidate(class_name: str, method_name: str) -> bool:
    """Filter heuristics: skip internal JVM methods."""
    skip_patterns = [
        r"^java/lang/",
        r"^sun/",
        r"^jdk/",
        r"Intrinsics$",
        r"\.class\$",
    ]
    for pattern in skip_patterns:
        if re.search(pattern, class_name):
            return False
    return True


class JfrHotspotProfiler:
    """Profile JVM hotspots via JFR ExecutionSample events."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jfr_collector = JfrCollector(output_dir=output_dir)

    def profile(self, pid: int, duration_sec: int = 30) -> HotspotProfile:
        """Profile JVM process to identify hotspot methods.

        Args:
            pid: JVM process ID
            duration_sec: Profiling duration in seconds

        Returns:
            HotspotProfile with identified hotspot methods
        """
        jdk_version = _detect_jdk_version(pid)

        # Start JFR recording with hotspot events
        jfr_file = self.output_dir / f"hotspot-{pid}.jfr"
        recording_name = "arkprobe-hotspot"

        events = JFR_HOTSPOT_PROFILES["profiles"]
        events_str = ",".join(events)

        start_cmd = [
            "jcmd", str(pid), "JFR.start",
            f"name={recording_name}",
            "settings=profile",
            f"duration={duration_sec}s",
            f"filename={jfr_file}",
            f"events={events_str}",
        ]

        log.info("Starting JFR hotspot profiling: %s", " ".join(start_cmd))
        start_result = run_cmd(start_cmd, timeout_sec=30)

        if not start_result.ok:
            log.warning("JFR.start failed, trying without custom events: %s", start_result.stderr[:200])
            # Fallback to default profile
            start_cmd = [
                "jcmd", str(pid), "JFR.start",
                f"name={recording_name}",
                "settings=profile",
                f"duration={duration_sec}s",
                f"filename={jfr_file}",
            ]
            start_result = run_cmd(start_cmd, timeout_sec=30)

        if not start_result.ok:
            log.error("JFR.start failed: %s", start_result.stderr[:200])
            return HotspotProfile(pid=pid, jdk_version=jdk_version, duration_sec=duration_sec)

        # Wait for profiling to complete
        import time
        log.info("Hotspot profiling in progress (%ds)...", duration_sec)
        time.sleep(duration_sec + 2)

        # Stop recording
        stop_cmd = ["jcmd", str(pid), "JFR.stop", f"name={recording_name}"]
        run_cmd(stop_cmd, timeout_sec=30)

        if not jfr_file.exists():
            log.error("JFR file not created: %s", jfr_file)
            return HotspotProfile(pid=pid, jdk_version=jdk_version, duration_sec=duration_sec)

        # Parse JFR recording
        try:
            methods = self._parse_jfr_recording(jfr_file, pid, duration_sec)
        except RuntimeError as e:
            log.warning("JFR parsing failed: %s", e)
            # Keep JFR file for debugging, return empty profile
            return HotspotProfile(
                pid=pid,
                jdk_version=jdk_version,
                duration_sec=duration_sec,
                methods=[],
                total_cpu_time_ns=0,
                jfr_file=jfr_file,
            )

        total_cpu_time = sum(m.cpu_time_ns for m in methods)

        profile = HotspotProfile(
            pid=pid,
            jdk_version=jdk_version,
            duration_sec=duration_sec,
            methods=methods,
            total_cpu_time_ns=total_cpu_time,
            jfr_file=jfr_file,
        )

        log.info("Hotspot profiling complete: %d methods identified", len(methods))
        return profile

    def _parse_jfr_recording(self, jfr_file: Path, pid: int, duration_sec: int = 30) -> list[HotspotMethod]:
        """Parse JFR recording to extract ExecutionSample events."""
        methods: dict[str, dict] = {}

        # Try to parse as JSON first
        print_result = run_cmd(["jfr", "print", "--json", "--events", "jdk.ExecutionSample", str(jfr_file)], timeout_sec=120)

        if print_result.ok and print_result.stdout:
            try:
                data = json.loads(print_result.stdout)
                events = data.get("events", [])
                for event in events:
                    self._process_execution_sample(event, methods)
            except json.JSONDecodeError:
                log.warning("Failed to parse JFR JSON output")

        # Fallback: parse text output
        if not methods:
            text_result = run_cmd(["jfr", "print", "--events", "jdk.ExecutionSample", str(jfr_file)], timeout_sec=120)
            if text_result.ok and text_result.stdout:
                self._parse_text_samples(text_result.stdout, methods)

        # Raise if no methods found — profiling produced no usable data
        if not methods:
            raise RuntimeError(
                f"JFR profiling yielded no hotspot methods. "
                f"Check that the target JVM was actively executing Java code "
                f"during the {duration_sec}s sampling window, "
                f"and that JDK's jfr tool is available."
            )

        # Convert to HotspotMethod objects
        hotspot_methods = []
        for key, stats in methods.items():
            parts = key.split("::", 2)
            if len(parts) != 3:
                log.warning("Skipping malformed method key: %s", key)
                continue
            class_name, method_name, signature = parts
            if _is_hotspot_candidate(class_name, method_name):
                hotspot_methods.append(HotspotMethod(
                    name=f"{class_name}.{method_name}",
                    signature=signature,
                    bytecode_size=stats.get("bytecode_size", 0),
                    compilation_count=stats.get("compilation_count", 0),
                    cpu_time_ns=stats.get("cpu_time_ns", 0),
                    inline_count=stats.get("inline_count", 0),
                    deopt_risk=stats.get("deopt_risk", 0.0),
                    simd_potential=stats.get("simd_potential", 0.0),
                    pattern_type=stats.get("pattern_type", "unknown"),
                ))

        return hotspot_methods

    def _process_execution_sample(self, event: dict, methods: dict) -> None:
        """Process a single ExecutionSample JFR event."""
        stack_trace = event.get("stackTrace", {})
        frames = stack_trace.get("frames", [])

        if not frames:
            return

        # Get top frame (innermost method)
        frame = frames[0]
        class_name, method_name, signature = _parse_stack_trace(frame)

        key = f"{class_name}::{method_name}::{signature}"

        if key not in methods:
            methods[key] = {
                "sample_count": 0,
                "cpu_time_ns": 0,
                "compilation_count": 0,
                "bytecode_size": 0,
                "inline_count": 0,
                "deopt_risk": 0.0,
                "simd_potential": 0.0,
                "pattern_type": "unknown",
            }

        methods[key]["sample_count"] += 1
        # Estimate CPU time: ~1ms per sample at 1000Hz sampling rate
        methods[key]["cpu_time_ns"] += 1_000_000

    def _parse_text_samples(self, text_output: str, methods: dict) -> None:
        """Parse JFR text output for ExecutionSample events."""
        for line in text_output.splitlines():
            line = line.strip()
            if not line:
                continue

            # Skip JDK internal frames
            if line.startswith("java.") or line.startswith("jdk."):
                continue

            # Try to extract class.method from text format
            match = re.search(r"([a-zA-Z0-9_$.]+)\.([a-zA-Z0-9_]+)\((.*?)\)", line)
            if match:
                class_name = match.group(1).replace("/", ".")
                method_name = match.group(2)
                signature = f"({match.group(3)})"

                key = f"{class_name}::{method_name}::{signature}"
                if key not in methods:
                    methods[key] = {
                        "sample_count": 0,
                        "cpu_time_ns": 0,
                    }
                methods[key]["sample_count"] += 1
                methods[key]["cpu_time_ns"] += 1_000_000
