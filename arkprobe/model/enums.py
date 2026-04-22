"""Enumerations for workload characterization."""

from enum import Enum


class ScenarioType(str, Enum):
    """Workload scenario categories."""
    DATABASE_OLTP = "database_oltp"
    DATABASE_OLAP = "database_olap"
    DATABASE_KV = "database_kv"
    BIGDATA_BATCH = "bigdata_batch"
    BIGDATA_STREAMING = "bigdata_streaming"
    CODEC_VIDEO = "codec_video"
    CODEC_AUDIO = "codec_audio"
    SEARCH_RECOMMEND = "search_recommend"
    MICROSERVICE = "microservice"
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    MIXED = "mixed"
    JVM_GENERAL = "jvm_general"
    JVM_GC_HEAVY = "jvm_gc_heavy"
    JVM_JIT_INTENSIVE = "jvm_jit_intensive"


class BottleneckCategory(str, Enum):
    """TopDown bottleneck classification."""
    FRONTEND_BOUND = "frontend_bound"
    BACKEND_MEMORY_BOUND = "backend_memory_bound"
    BACKEND_CORE_BOUND = "backend_core_bound"
    BAD_SPECULATION = "bad_speculation"
    WELL_BALANCED = "well_balanced"
    JVM_GC_PAUSE = "jvm_gc_pause"
    JVM_SAFEPOINT = "jvm_safepoint"
    JVM_JIT_DEOPT = "jvm_jit_deopt"
    JVM_HEAP_PRESSURE = "jvm_heap_pressure"


class AccessPattern(str, Enum):
    """Memory access pattern types."""
    STRIDE = "stride"
    RANDOM = "random"
    STREAMING = "streaming"
    MIXED = "mixed"


class IPCMechanism(str, Enum):
    """Inter-process communication mechanisms."""
    SHARED_MEMORY = "shared_memory"
    SOCKET = "socket"
    PIPE = "pipe"
    NONE = "none"


class PrefetchLevel(str, Enum):
    """Hardware prefetcher aggressiveness levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class TuningLayer(str, Enum):
    """Platform tuning parameter layers."""
    OS = "os"
    BIOS = "bios"
    DRIVER = "driver"
    JVM = "jvm"


class TuningDifficulty(str, Enum):
    """How difficult/risky to apply a tuning change."""
    TRIVIAL = "trivial"      # sysctl, echo to /sys — no restart needed
    EASY = "easy"            # service restart needed
    MODERATE = "moderate"    # reboot needed (GRUB params, some BIOS)
    HARD = "hard"            # BIOS change, firmware update


class TuningRiskLevel(str, Enum):
    """Risk of applying the tuning change in production."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
