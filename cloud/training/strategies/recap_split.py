from __future__ import annotations


class RECAPSplitTrainingStrategy:
    name = "recap_split"

    def __init__(self, continual_learner) -> None:
        self.continual_learner = continual_learner

    def train_from_workspace(
        self,
        workspace: str,
        *,
        edge_id: int,
        base_model_version: str = "0",
        result_model_version: str = "1",
    ):
        del base_model_version, result_model_version
        return self.continual_learner.get_ground_truth_and_fixed_split_retrain(
            int(edge_id),
            workspace,
        )
