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
TEMPLATES_DIR = Path(__file__).parent / "templates"


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

    platform = PlatformConfig.model_validate(raw.get("platform", {}))
    workload = WorkloadConfig.model_validate(raw.get("workload", {}))
    collection = CollectionConfig.model_validate(raw.get("collection", {}))
    scalability = ScalabilityConfig.model_validate(raw.get("scalability", {}))

    focus = raw.get("focus_metrics", [])
    if isinstance(focus, dict):
        focus_metrics = focus.get("metrics", [])
        focus_desc = focus.get("description", "")
    elif isinstance(focus, list):
        focus_metrics = focus
        focus_desc = ""
    else:
        focus_metrics = []
        focus_desc = ""

    return ScenarioConfig(
        name=name,
        type=ScenarioType(scenario_type),
        description=description,
        platform=platform,
        workload=workload,
        collection=collection,
        scalability=scalability,
        focus_metrics=focus_metrics,
        focus_description=focus_desc,
    )


def load_all_scenarios(configs_dir: Optional[Path] = None) -> List[ScenarioConfig]:
    """Load all scenario configs from the configs directory."""
    if configs_dir is None:
        configs_dir = CONFIGS_DIR

    scenarios = []
    if not configs_dir.exists():
        log.warning("Scenario configs directory not found: %s", configs_dir)
        return scenarios

    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        try:
            scenario = load_scenario(yaml_file)
            scenarios.append(scenario)
            log.info("Loaded scenario: %s (%s)", scenario.name, scenario.type)
        except Exception as e:
            log.error("Failed to load %s: %s", yaml_file.name, e)

    return scenarios


def list_scenarios(configs_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """List available scenarios with name and type."""
    scenarios = load_all_scenarios(configs_dir)
    return [{"name": s.name, "type": s.type.value, "file": ""} for s in scenarios]


def get_scenario_by_name(name: str, configs_dir: Optional[Path] = None) -> Optional[ScenarioConfig]:
    """Find a scenario by name."""
    for s in load_all_scenarios(configs_dir):
        if s.name == name or s.name.lower().replace(" ", "_") == name.lower().replace(" ", "_"):
            return s
    return None
