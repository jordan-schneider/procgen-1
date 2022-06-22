from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, cast

import numpy as np

DataModality = Literal["state", "action", "traj"]
QuestionAlgorithm = Literal["random", "infogain", "manual"]


def assure_modality(modality: str) -> DataModality:
    if not (modality == "state" or modality == "action" or modality == "traj"):
        raise ValueError(f"Unknown modality: {modality}")
    modality = cast(DataModality, modality)
    return modality


@dataclass
class State:
    grid: np.ndarray
    grid_shape: Tuple[int, int]
    agent_pos: Tuple[int, int]
    exit_pos: Tuple[int, int]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, State)
            and np.array_equal(self.grid, other.grid)
            and self.grid_shape == other.grid_shape
            and self.agent_pos == other.agent_pos
            and self.exit_pos == other.exit_pos
        )

    @staticmethod
    def from_json(json_dict: dict) -> State:
        grid = np.array(list(json_dict["grid"].values()))
        grid_shape = json_dict["grid_shape"]
        agent_pos = json_dict["agent_pos"]
        exit_pos = json_dict["exit_pos"]
        return State(grid, grid_shape, agent_pos, exit_pos)


@dataclass
class Trajectory:
    start_state: State
    actions: Optional[np.ndarray]
    env_name: str
    modality: DataModality

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Trajectory)
            and self.start_state == other.start_state
            and np.array_equal(self.actions, other.actions)  # type: ignore
            and self.env_name == other.env_name
            and self.modality == other.modality
        )


@dataclass
class FeatureTrajectory(Trajectory):
    features: np.ndarray

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FeatureTrajectory)
            and super().__eq__(other)
            and np.array_equal(self.features, other.features)
        )


@dataclass
class Question:
    id: int
    trajs: Tuple[Trajectory, Trajectory]


@dataclass
class Answer:
    user_id: int
    question_id: int
    answer: bool
    start_time: str
    end_time: str


# TODO: Decide what demographics might be interesting
@dataclass
class Demographics:
    age: int
