from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineResult:
    success: bool
    model_data: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[bool, str, str]:
        return self.success, self.model_data, self.message


@dataclass(frozen=True)
class SampleSyncResult:
    success: bool
    message: str
    committed_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[bool, str, int]:
        return self.success, self.message, self.committed_samples
