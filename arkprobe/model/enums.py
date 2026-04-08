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


class BottleneckCategory(str, Enum):
    """TopDown bottleneck classification."""
    FRONTEND_BOUND = "frontend_bound"
    BACKEND_MEMORY_BOUND = "backend_memory_bound"
    BACKEND_CORE_BOUND = "backend_core_bound"
    BAD_SPECULATION = "bad_speculation"
    WELL_BALANCED = "well_balanced"


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
