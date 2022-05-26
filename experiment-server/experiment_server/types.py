from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

DataModality = Literal["state", "action", "traj"]


@dataclass
class State:
    grid: np.ndarray
    agent_pos: Tuple[int, int]
    exit_pos: Tuple[int, int]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, State)
            and np.array_equal(self.grid, other.grid)
            and self.agent_pos == other.agent_pos
            and self.exit_pos == other.exit_pos
        )


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


Question = Tuple[Trajectory, Trajectory]
