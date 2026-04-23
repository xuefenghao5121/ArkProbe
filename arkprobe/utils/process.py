"""Subprocess execution helpers with timeout and output capture."""
import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)

@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    command: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

def run_cmd(
    cmd: List[str],
    timeout_sec: int = 300,
    check: bool = False,
    capture: bool = True,
    cwd: Optional[str] = None,
) -> RunResult:
    """Run a command with timeout and structured output."""
    cmd_str = " ".join(cmd)
    log.debug("Running: %s", cmd_str)
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout_sec,
            cwd=cwd,
        )
        run_result = RunResult(
            returncode=result.returncode,
            stdout=result.stdout if capture else "",
            stderr=result.stderr if capture else "",
            command=cmd_str,
        )
        if check and not run_result.ok:
            log.error("Command failed (rc=%d): %s\nstderr: %s",
                      result.returncode, cmd_str, run_result.stderr[:500])
            raise subprocess.CalledProcessError(result.returncode, cmd_str)
        return run_result
    except subprocess.TimeoutExpired:
        log.error("Command timed out after %ds: %s", timeout_sec, cmd_str)
        return RunResult(returncode=-1, stdout="", stderr=f"Timeout after {timeout_sec}s", command=cmd_str)

def run_shell(
    cmd: str,
    timeout_sec: int = 300,
    check: bool = False,
) -> RunResult:
    """Run a shell command string.

    WARNING: Prefer run_cmd() with list-form arguments to avoid shell injection.
    Only use this when shell features (pipes, redirects) are truly needed.
    """
    log.warning("run_shell() called — prefer run_cmd() with list-form args to avoid injection risk")
    log.debug("Running shell: %s", cmd)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout_sec,
        )
        return RunResult(
            returncode=result.returncode, stdout=result.stdout,
            stderr=result.stderr, command=cmd,
        )
    except subprocess.TimeoutExpired:
        return RunResult(returncode=-1, stdout="", stderr=f"Timeout after {timeout_sec}s", command=cmd)
