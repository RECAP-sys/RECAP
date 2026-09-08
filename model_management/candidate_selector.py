from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from model_management.split_candidate import CandidateProfile, SplitCandidate


def _profile_by_id(
    candidates: Sequence[SplitCandidate], profiles: Sequence[CandidateProfile]
) -> dict[str, CandidateProfile]:
    profile_map = {profile.candidate_id: profile for profile in profiles}
    for candidate in candidates:
        profile_map.setdefault(
            candidate.candidate_id,
            CandidateProfile(
                candidate_id=candidate.candidate_id,
                edge_flops=candidate.estimated_edge_flops,
                cloud_flops=candidate.estimated_cloud_flops,
                payload_bytes=candidate.estimated_payload_bytes,
                boundary_tensor_count=candidate.boundary_count,
                boundary_shape_summary=[],
                estimated_privacy_leakage=candidate.estimated_privacy_risk,
                measured_edge_latency=0.0,
                measured_cloud_latency=0.0,
                measured_end_to_end_latency=candidate.estimated_latency,
                replay_success_rate=1.0 if candidate.is_validated else 0.0,
                tail_trainability=candidate.is_trainable_tail,
                stability_score=1.0 if candidate.is_validated else 0.0,
                validation_passed=candidate.is_validated,
            ),
        )
    return profile_map


@dataclass
class SelectorState:
    A: np.ndarray
    b: np.ndarray
    last_context: np.ndarray | None = None
    historical_reward: float = 0.0
    num_updates: int = 0
    invalidated: bool = False


class SplitCandidateSelector:
    def __init__(
        self,
        candidates: Sequence[SplitCandidate],
        profiles: Sequence[CandidateProfile] | None = None,
        *,
        alpha: float = 0.35,
        epsilon: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.epsilon = epsilon
        self.candidates = {candidate.candidate_id: candidate for candidate in candidates}
        self.profiles = _profile_by_id(candidates, profiles or [])
        self.states: dict[str, SelectorState] = {}
        self.feature_dim = 13
        for candidate in candidates:
            self.states[candidate.candidate_id] = SelectorState(
                A=np.eye(self.feature_dim, dtype=np.float64),
                b=np.zeros(self.feature_dim, dtype=np.float64),
            )

    def fit_context(
        self,
        candidate_id: str,
        *,
        bandwidth: float = 1.0,
        edge_load: float = 0.0,
        cloud_load: float = 0.0,
    ) -> np.ndarray:
        candidate = self.candidates[candidate_id]
        profile = self.profiles[candidate_id]
        state = self.states[candidate_id]
        context = np.array(
            [
                float(profile.edge_flops),
                float(profile.cloud_flops),
                float(profile.payload_bytes),
                float(profile.boundary_tensor_count),
                float(profile.estimated_privacy_leakage),
                float(profile.measured_end_to_end_latency or candidate.estimated_latency),
                float(bandwidth),
                float(edge_load),
                float(cloud_load),
                1.0 if profile.validation_passed else 0.0,
                float(profile.stability_score),
                1.0 if profile.tail_trainability else 0.0,
                float(state.historical_reward),
            ],
            dtype=np.float64,
        )
        denom = np.maximum(np.abs(context).max(), 1.0)
        context = context / denom
        state.last_context = context
        return context

    def _heuristic_score(self, candidate_id: str) -> float:
        candidate = self.candidates[candidate_id]
        profile = self.profiles[candidate_id]
        validation_penalty = 5.0 if not profile.validation_passed else 0.0
        stability_penalty = 1.0 - profile.stability_score
        return (
            -float(profile.measured_end_to_end_latency or candidate.estimated_latency)
            - 0.5 * float(profile.payload_bytes) / float(1024 * 1024)
            - 0.25 * float(profile.estimated_privacy_leakage)
            - stability_penalty
            - validation_penalty
            + (0.5 if profile.tail_trainability else 0.0)
        )

    def select_candidate(
        self,
        *,
        bandwidth: float = 1.0,
        edge_load: float = 0.0,
        cloud_load: float = 0.0,
        require_trainable_tail: bool = False,
    ) -> str:
        valid_ids = [
            candidate_id
            for candidate_id, candidate in self.candidates.items()
            if not self.states[candidate_id].invalidated
            and (not require_trainable_tail or candidate.is_trainable_tail)
        ]
        if not valid_ids:
            raise RuntimeError("No valid split candidates remain.")

        scores: list[tuple[float, str]] = []
        for candidate_id in valid_ids:
            state = self.states[candidate_id]
            context = self.fit_context(
                candidate_id,
                bandwidth=bandwidth,
                edge_load=edge_load,
                cloud_load=cloud_load,
            )
            try:
                inv = np.linalg.inv(state.A)
                theta = inv @ state.b
                bonus = self.alpha * float(np.sqrt(context.T @ inv @ context))
                score = float(theta.T @ context) + bonus
            except np.linalg.LinAlgError:
                score = self._heuristic_score(candidate_id)
            score += 0.05 * state.historical_reward
            scores.append((score, candidate_id))

        scores.sort(reverse=True)
        shortlist = scores[: max(1, min(3, len(scores)))]
        best_score, best_id = shortlist[0]
        if self.epsilon > 0.0 and len(shortlist) > 1:
            threshold = max(1, int(round(1.0 / self.epsilon)))
            total_updates = sum(self.states[candidate_id].num_updates for candidate_id in valid_ids)
            if threshold > 0 and total_updates % threshold == 0:
                return shortlist[-1][1]
        return best_id

    def update_reward(
        self,
        candidate_id: str,
        reward: float,
        *,
        context: np.ndarray | None = None,
    ) -> None:
        state = self.states[candidate_id]
        x = context if context is not None else state.last_context
        if x is None:
            return
        state.A = state.A + np.outer(x, x)
        state.b = state.b + reward * x
        state.historical_reward = 0.8 * state.historical_reward + 0.2 * reward
        state.num_updates += 1

    def invalidate_candidate(self, candidate_id: str) -> None:
        if candidate_id in self.states:
            self.states[candidate_id].invalidated = True

    def cache_profile(self, profile: CandidateProfile) -> None:
        self.profiles[profile.candidate_id] = profile
