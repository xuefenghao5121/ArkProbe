"""Dependency checker — detect whether scenario workload binaries are available."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from .registry import get_install_hint


@dataclass
class DepCheckResult:
    """Result of checking a single binary dependency."""
    binary: str
    available: bool
    path: str
    install_hint: str


def check_binary(name: str) -> DepCheckResult:
    """Check if a binary is available on PATH."""
    found = shutil.which(name)
    return DepCheckResult(
        binary=name,
        available=found is not None,
        path=found or "",
        install_hint=get_install_hint(name),
    )


def check_dependencies(dep_list: list[str]) -> list[DepCheckResult]:
    """Check a list of binary dependencies."""
    return [check_binary(name) for name in dep_list]


def check_all_available(dep_list: list[str]) -> bool:
    """Return True if all dependencies are available."""
    return all(check_binary(name).available for name in dep_list)


def format_missing_deps(results: list[DepCheckResult]) -> str:
    """Format missing dependencies as a human-readable string."""
    missing = [r for r in results if not r.available]
    if not missing:
        return ""
    lines = ["Missing dependencies:"]
    for r in missing:
        lines.append(f"  - {r.binary}: {r.install_hint}")
    return "\n".join(lines)
