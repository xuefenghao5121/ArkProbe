"""
Shared data models for the hotspot module.

These models are imported by all submodules, so they're defined
here to avoid circular import issues.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HotspotMethod:
    """Represents a JVM hotspot method identified via JFR profiling."""
    name: str                      # Fully qualified class name: "com.example.FastMath.compute"
    signature: str = ""            # JVM method signature: "(II)D"
    bytecode_size: int = 0        # Size of bytecode in bytes
    compilation_count: int = 0    # Number of times JIT compiled this method
    cpu_time_ns: int = 0          # Estimated CPU time from ExecutionSample events (nanoseconds)
    cpu_time_percent: float = 0.0  # CPU time as percentage of total (set from total)
    inline_count: int = 0         # How many times this method was inlined
    deopt_risk: float = 0.0       # Deoptimization risk score [0.0, 1.0]
    simd_potential: float = 0.0   # SIMD acceleration potential [0.0, 1.0]
    pattern_type: str = "unknown" # "vector_expr" | "string" | "math" | "unknown"
    bytecode_hex: Optional[str] = None  # Raw bytecode in hex format for analysis

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "signature": self.signature,
            "bytecode_size": self.bytecode_size,
            "compilation_count": self.compilation_count,
            "cpu_time_ns": self.cpu_time_ns,
            "cpu_time_percent": self.cpu_time_percent,
            "inline_count": self.inline_count,
            "deopt_risk": self.deopt_risk,
            "simd_potential": self.simd_potential,
            "pattern_type": self.pattern_type,
            "bytecode_hex": self.bytecode_hex,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HotspotMethod":
        """Create HotspotMethod from dictionary with explicit field extraction."""
        return cls(
            name=str(data.get("name", "")),
            signature=str(data.get("signature", "")),
            bytecode_size=int(data.get("bytecode_size", 0)),
            compilation_count=int(data.get("compilation_count", 0)),
            cpu_time_ns=int(data.get("cpu_time_ns", 0)),
            cpu_time_percent=float(data.get("cpu_time_percent", 0.0)),
            inline_count=int(data.get("inline_count", 0)),
            deopt_risk=float(data.get("deopt_risk", 0.0)),
            simd_potential=float(data.get("simd_potential", 0.0)),
            pattern_type=str(data.get("pattern_type", "unknown")),
            bytecode_hex=data.get("bytecode_hex"),
        )


@dataclass
class HotspotProfile:
    """Collection of hotspot methods identified from JFR profiling."""
    pid: int
    jdk_version: str
    duration_sec: int
    methods: list[HotspotMethod] = field(default_factory=list)
    total_cpu_time_ns: int = 0
    jfr_file: Optional[str] = None

    def get_top_methods(self, limit: int = 10) -> list[HotspotMethod]:
        """Return top N methods by CPU time."""
        return sorted(self.methods, key=lambda m: m.cpu_time_ns, reverse=True)[:limit]

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "jdk_version": self.jdk_version,
            "duration_sec": self.duration_sec,
            "total_methods": len(self.methods),
            "total_cpu_time_ns": self.total_cpu_time_ns,
            "top_methods": [m.to_dict() for m in self.get_top_methods()],
        }
