"""YAML scenario configuration loader and validator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from ..model.enums import ScenarioType

log = logging.getLogger(__name__)

# Default configs directory
CONFIGS_DIR = Path(__file__).parent / "configs"
BUILTIN_DIR = Path(__file__).parent / "builtin"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Explicit short name -> YAML filename mapping for builtin scenarios.
# Provides unambiguous resolution regardless of external scenario names.
BUILTIN_SHORT_NAMES: Dict[str, str] = {
    "compute": "compute_intensive.yaml",
    "memory": "memory_intensive.yaml",
    "mixed": "mixed_workload.yaml",
    "stream": "stream_bandwidth.yaml",
    "random": "random_access.yaml",
    "crypto": "crypto.yaml",
    "compress": "compress.yaml",
    "video": "video_encoding.yaml",
    "ml": "ml_inference.yaml",
    "oltp": "database_oltp.yaml",
    "kv": "kv_store.yaml",
    "web": "web_server.yaml",
}


class PlatformConfig(BaseModel):
    kunpeng_model: str = "920"
    min_cores: int = 4
    recommended_cores: int = 64


class WorkloadConfig(BaseModel):
    setup: List[str] = Field(default_factory=list)
    command: str
    teardown: List[str] = Field(default_factory=list)
    target_process: str = ""
    throughput_metric: str = "ops/sec"
    throughput_regex: str = r'(\d+\.?\d*)\s+ops/sec'


class CollectionConfig(BaseModel):
    perf_duration_sec: int = 60
    ebpf_duration_sec: int = 30
    warmup_sec: int = 30
    event_groups: List[str] = Field(default_factory=lambda: [
        "topdown_l1", "instruction_mix", "cache_l1",
        "cache_l2_l3", "branch_prediction", "memory_access",
    ])
    ebpf_probes: List[str] = Field(default_factory=lambda: [
        "io_latency", "lock_contention", "cache_stats", "tcp_latency",
        "mem_access", "sched_latency",
    ])


class ScalabilityConfig(BaseModel):
    enabled: bool = False
    core_counts: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 48, 64])
    thread_counts: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 48, 64])


class ScenarioConfig(BaseModel):
    """Validated scenario configuration."""
    name: str
    type: ScenarioType
    description: str = ""
    builtin: bool = False
    dependencies: List[str] = Field(default_factory=list)
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    workload: WorkloadConfig
    collection: CollectionConfig = Field(default_factory=CollectionConfig)
    scalability: ScalabilityConfig = Field(default_factory=ScalabilityConfig)
    focus_metrics: List[str] = Field(default_factory=list)
    focus_description: str = ""


def load_scenario(path: Path) -> ScenarioConfig:
    """Load and validate a single scenario YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    scenario_data = raw.get("scenario", {})
    name = scenario_data.get("name", path.stem)
    scenario_type = scenario_data.get("type", "microservice")
    description = scenario_data.get("description", "")
    builtin = scenario_data.get("builtin", False)

    dependencies = raw.get("dependencies", [])
    if dependencies is None:
        dependencies = []

    platform = PlatformConfig.model_validate(raw.get("platform", {}))
    workload = WorkloadConfig.model_validate(raw.get("workload", {}))
    collection = CollectionConfig.model_validate(raw.get("collection", {}))
    scalability = ScalabilityConfig.model_validate(raw.get("scalability", {}))

    focus = raw.get("focus_metrics", [])
    if isinstance(focus, dict):
        focus_metrics = focus.get("metrics", [])
        focus_desc = focus.get("description", "")
    elif isinstance(focus, list):
        # focus_metrics entries can be strings or dicts with 'name' key
        focus_metrics = []
        for item in focus:
            if isinstance(item, str):
                focus_metrics.append(item)
            elif isinstance(item, dict) and "name" in item:
                focus_metrics.append(item["name"])
        focus_desc = ""
    else:
        focus_metrics = []
        focus_desc = ""

    return ScenarioConfig(
        name=name,
        type=ScenarioType(scenario_type),
        description=description,
        builtin=builtin,
        dependencies=dependencies,
        platform=platform,
        workload=workload,
        collection=collection,
        scalability=scalability,
        focus_metrics=focus_metrics,
        focus_description=focus_desc,
    )


def load_all_scenarios(
    configs_dir: Optional[Path] = None,
    include_builtin: bool = True,
) -> List[ScenarioConfig]:
    """Load all scenario configs from configs and builtin directories."""
    if configs_dir is None:
        configs_dir = CONFIGS_DIR

    scenarios = []
    dirs = [configs_dir]
    if include_builtin and BUILTIN_DIR.exists():
        dirs.append(BUILTIN_DIR)

    for scan_dir in dirs:
        if not scan_dir.exists():
            log.warning("Scenario configs directory not found: %s", scan_dir)
            continue
        for yaml_file in sorted(scan_dir.glob("*.yaml")):
            try:
                scenario = load_scenario(yaml_file)
                scenarios.append(scenario)
                log.info("Loaded scenario: %s (%s)", scenario.name, scenario.type)
            except Exception as e:
                log.error("Failed to load %s: %s", yaml_file.name, e)

    return scenarios


def load_builtin_scenarios() -> List[ScenarioConfig]:
    """Load only builtin scenarios (zero external dependencies)."""
    if not BUILTIN_DIR.exists():
        return []
    scenarios = []
    for yaml_file in sorted(BUILTIN_DIR.glob("*.yaml")):
        try:
            scenario = load_scenario(yaml_file)
            scenarios.append(scenario)
        except Exception as e:
            log.error("Failed to load builtin %s: %s", yaml_file.name, e)
    return scenarios


def list_scenarios_lightweight(
    configs_dir: Optional[Path] = None,
    include_builtin: bool = True,
) -> List[Dict[str, Any]]:
    """List scenarios by reading only YAML metadata, no pydantic validation.

    This never raises errors for malformed YAML — it skips bad files silently.
    """
    if configs_dir is None:
        configs_dir = CONFIGS_DIR

    results: List[Dict[str, Any]] = []
    dirs = [configs_dir]
    if include_builtin and BUILTIN_DIR.exists():
        dirs.append(BUILTIN_DIR)

    for scan_dir in dirs:
        if not scan_dir.exists():
            continue
        for yaml_file in sorted(scan_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    continue
                scenario_data = raw.get("scenario", {})
                results.append({
                    "name": scenario_data.get("name", yaml_file.stem),
                    "type": scenario_data.get("type", "unknown"),
                    "description": scenario_data.get("description", ""),
                    "builtin": scenario_data.get("builtin", False),
                    "dependencies": raw.get("dependencies", []) or [],
                    "file": str(yaml_file),
                })
            except Exception:
                # Skip unreadable files — don't block the listing
                continue

    return results


def list_scenarios(configs_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """List available scenarios with name and type."""
    scenarios = load_all_scenarios(configs_dir)
    return [{"name": s.name, "type": s.type.value, "builtin": s.builtin, "file": ""} for s in scenarios]


def get_scenario_by_name(name: str, configs_dir: Optional[Path] = None) -> Optional[ScenarioConfig]:
    """Find a scenario by name. Supports 'builtin' as a special group name.

    Resolution order:
    1. Builtin short name (compute/memory/mixed/etc.) -> exact YAML file
    2. Exact name match across all scenarios
    3. Fuzzy substring match (case/space/underscore/hyphen insensitive)
    """
    # 1. Check builtin short name mapping first (highest priority)
    yaml_file = BUILTIN_SHORT_NAMES.get(name.lower())
    if yaml_file and BUILTIN_DIR.exists():
        path = BUILTIN_DIR / yaml_file
        if path.exists():
            try:
                return load_scenario(path)
            except Exception:
                pass

    # Normalize input for fuzzy matching
    name_clean = name.lower().replace("_", "").replace("-", "").replace(" ", "")

    # 2. Exact match, then fuzzy substring match
    # Prefer builtin scenarios when ambiguous
    all_scenarios = load_all_scenarios(configs_dir)

    # Exact match first
    for s in all_scenarios:
        if s.name == name or s.name.lower().replace(" ", "_") == name.lower().replace(" ", "_"):
            return s

    # Fuzzy substring match — prefer builtin scenarios
    builtin_match = None
    other_match = None
    for s in all_scenarios:
        s_clean = s.name.lower().replace("_", "").replace("-", "").replace(" ", "")
        if name_clean in s_clean or s_clean in name_clean:
            if s.builtin:
                builtin_match = s
            else:
                other_match = s

    return builtin_match or other_match
