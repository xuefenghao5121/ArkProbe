"""Unit conversion utilities for performance metrics."""

def mpki(event_count: int, instructions: int) -> float:
    """Compute Misses Per Kilo-Instructions."""
    if instructions <= 0:
        return 0.0
    return (event_count / instructions) * 1000.0

def miss_rate(misses: int, accesses: int) -> float:
    """Compute miss rate as a fraction 0..1."""
    if accesses <= 0:
        return 0.0
    return min(misses / accesses, 1.0)

def bytes_to_gbps(byte_count: int, duration_sec: float) -> float:
    """Convert byte count over duration to GB/s."""
    if duration_sec <= 0:
        return 0.0
    return byte_count / (duration_sec * 1e9)

def bytes_to_mbps(byte_count: int, duration_sec: float) -> float:
    """Convert byte count over duration to MB/s."""
    if duration_sec <= 0:
        return 0.0
    return byte_count / (duration_sec * 1e6)

def format_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))
