"""
Bytecode extractor for retrieving method bytecode from JVM processes.

Uses jcmd HSDB (HotSpot Debugger) and class histogram to extract bytecode.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from arkprobe.utils.process import run_cmd

log = logging.getLogger(__name__)


class BytecodeExtractor:
    """Extract bytecode from running JVM processes."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_class_histogram(self, pid: int) -> dict:
        """Get class histogram from JVM process (memory footprint per class).

        Returns:
            Dict mapping class names to instance counts
        """
        cmd = ["jcmd", str(pid), "GC.class_histogram"]
        result = run_cmd(cmd, timeout_sec=60)

        histogram = {}
        if result.ok:
            lines = result.stdout.splitlines()
            for line in lines:
                # Format: "num #instances   #bytes class name"
                match = re.match(r"\s*\d+\s+(\d+)\s+(\d+)\s+([\w.$]+)", line)
                if match:
                    instances = int(match.group(1))
                    bytes_ = int(match.group(2))
                    class_name = match.group(3).replace("/", ".")
                    histogram[class_name] = {"instances": instances, "bytes": bytes_}

        log.info("Class histogram: %d classes found", len(histogram))
        return histogram

    def get_method_bytecode(self, pid: int, class_name: str, method_name: str) -> Optional[str]:
        """Get bytecode hex dump for a specific method.

        Args:
            pid: JVM process ID
            class_name: Fully qualified class name (e.g., "com.example.FastMath")
            method_name: Method name (e.g., "compute")

        Returns:
            Hex string of bytecode or None if not found
        """
        # Get class info via jcmd
        cmd = ["jcmd", str(pid), "VM.info"]
        result = run_cmd(cmd, timeout_sec=30)

        if not result.ok:
            log.error("Failed to get VM.info for PID %d", pid)
            return None

        # Try to disassemble class using javap if available
        javap_result = self._try_javap_disassemble(class_name, method_name)
        if javap_result:
            return javap_result

        # Fallback: try to extract from hsdb
        hsdb_bytecode = self._try_hsdb_extract(pid, class_name, method_name)
        if hsdb_bytecode:
            return hsdb_bytecode

        log.warning("Could not extract bytecode for %s.%s", class_name, method_name)
        return None

    def _try_javap_disassemble(self, class_name: str, method_name: str) -> Optional[str]:
        """Try to use javap to disassemble a class file."""
        # Try to find the .class file in common locations
        possible_paths = [
            f"/tmp/classes/{class_name.replace('.', '/')}.class",
            f"./target/classes/{class_name.replace('.', '/')}.class",
            f"./build/classes/java/main/{class_name.replace('.', '/')}.class",
        ]

        for class_path in possible_paths:
            if Path(class_path).exists():
                cmd = ["javap", "-c", "-p", class_path, method_name]
                result = run_cmd(cmd, timeout_sec=10)
                if result.ok:
                    log.info("Found bytecode via javap: %s", class_path)
                    return result.stdout

        return None

    def _try_hsdb_extract(self, pid: int, class_name: str, method_name: str) -> Optional[str]:
        """Try to extract bytecode using HSDB (HotSpot Debugger).

        Note: HSDB requires SA (Serviceability Agent) and interactive use.
        This is a placeholder for future implementation using jhsdb clhsdb.
        """
        # jhsdb clhsdb supports non-interactive commands in newer JDKs:
        #   jhsdb clhsdb --pid <pid> --command "bytecode <class>.<method>"
        # Not widely available yet, so we skip for now.

        log.debug("HSDB extraction not implemented; skipping")
        return None

    def get_bytecode_size(self, pid: int, class_name: str, method_name: str) -> int:
        """Estimate bytecode size for a method."""
        bytecode = self.get_method_bytecode(pid, class_name, method_name)
        if bytecode:
            # Count bytecode instructions (lines starting with numbers)
            lines = bytecode.splitlines()
            instruction_count = sum(1 for line in lines if re.match(r"\s+\d+:", line))
            # Rough estimate: ~1-2 bytes per instruction
            return instruction_count * 2
        return 0
