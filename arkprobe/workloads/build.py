"""Build and resolve built-in C workloads.

Compiles C sources on first use, caches binaries to ~/.cache/arkprobe/bin/.
Falls back to Python implementations if gcc is not available.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent / "src"
CACHE_DIR = Path.home() / ".cache" / "arkprobe" / "bin"

WORKLOAD_SOURCES = {
    "compute": "compute.c",
    "memory": "memory.c",
    "mixed": "mixed.c",
    "stream": "stream.c",
    "random": "random.c",
}

BINARY_PREFIX = "arkprobe_"


def _get_binary_path(name: str) -> Path:
    return CACHE_DIR / f"{BINARY_PREFIX}{name}"


def _compile_one(name: str, src_file: str) -> Path | None:
    """Compile a single C source file. Returns binary path or None on failure."""
    src = SRC_DIR / src_file
    if not src.exists():
        log.error("Source file not found: %s", src)
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _get_binary_path(name)

    # Skip if already compiled and source is older
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        return out

    gcc = shutil.which("gcc")
    if gcc is None:
        log.warning("gcc not found, cannot compile builtin workload '%s'", name)
        return None

    cmd = [gcc, "-O2", "-pthread", "-lm", "-o", str(out), str(src)]

    # Use -march=native only on matching architecture
    arch = platform.machine()
    if arch == "aarch64":
        cmd.insert(1, "-march=armv8-a")
    elif arch == "x86_64":
        cmd.insert(1, "-march=native")

    log.info("Compiling %s: %s", name, " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.error("Compilation failed for %s:\n%s", name, result.stderr)
            return None
        return out
    except Exception as e:
        log.error("Compilation error for %s: %s", name, e)
        return None


def build_workloads() -> dict[str, Path]:
    """Compile all C workloads. Returns {name: binary_path} for successful builds."""
    results = {}
    for name, src in WORKLOAD_SOURCES.items():
        path = _compile_one(name, src)
        if path is not None:
            results[name] = path
    return results


def get_workload_binary(name: str) -> Path | None:
    """Get path to a compiled workload binary, compiling if needed."""
    # Check cached first
    cached = _get_binary_path(name)
    if cached.exists():
        return cached

    # Try to compile
    src = WORKLOAD_SOURCES.get(name)
    if src is None:
        return None
    return _compile_one(name, src)


def resolve_builtin_command(command_template: str) -> str:
    """Replace {builtin_binary_XXX} placeholders with actual binary paths.

    Falls back to Python fallback module if C binary is not available.
    """
    import re
    pattern = r"\{builtin_binary_(\w+)\}"

    def replacer(match: re.Match) -> str:
        name = match.group(1)
        binary = get_workload_binary(name)
        if binary is not None:
            return str(binary)
        # Fallback to Python
        log.warning(
            "C binary for '%s' not available, using Python fallback. "
            "PMU data may include Python runtime overhead. "
            "Install gcc to compile native workloads.",
            name,
        )
        return f"python -m arkprobe.workloads.fallback --workload {name}"

    return re.sub(pattern, replacer, command_template)
