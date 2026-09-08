from __future__ import annotations

from typing import Protocol


class CloudTrainingStrategy(Protocol):
    name: str

    def train_from_workspace(
        self,
        workspace: str,
        *,
        base_model_version: str = "0",
        result_model_version: str = "1",
    ) -> dict[str, object]: ...
