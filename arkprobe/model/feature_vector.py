"""Feature vector serialization and I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import WorkloadFeatureVector


def save_feature_vector(fv: WorkloadFeatureVector, path: Path) -> None:
    """Serialize a feature vector to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fv.model_dump_json(indent=2), encoding="utf-8")


def load_feature_vector(path: Path) -> WorkloadFeatureVector:
    """Deserialize a feature vector from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return WorkloadFeatureVector.model_validate(data)


def load_feature_vectors(paths: List[Path]) -> List[WorkloadFeatureVector]:
    """Load multiple feature vectors."""
    return [load_feature_vector(p) for p in paths]
