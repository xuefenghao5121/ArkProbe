"""Abstract base class for all data collectors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class CollectionResult:
    """Result from a single collector run."""
    collector_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    raw_files: Dict[str, Path] = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


class BaseCollector(ABC):
    """Abstract base for all data collectors."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def collect(self, **kwargs) -> CollectionResult:
        """Run collection and return structured results."""
        ...

    def _save_raw(self, name: str, content: str) -> Path:
        """Save raw output to a file in the output directory."""
        path = self.output_dir / name
        path.write_text(content, encoding="utf-8")
        return path
