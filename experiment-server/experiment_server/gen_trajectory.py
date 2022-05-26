from dataclasses import dataclass
from typing import Callable, List, Tuple

import gym3  # type: ignore
import numpy as np
from linear_procgen import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen.feature_envs import FeatureEnv
from linear_procgen.util import get_root_env
from procgen.env import ENV_NAMES_T

from experiment_server.util import remove_duplicates, remove_zeros


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
    actions: np.ndarray
    env_name: str

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Trajectory)
            and self.start_state == other.start_state
            and np.array_equal(self.actions, other.actions)
            and self.env_name == other.env_name
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


def collect_trajs(
    env: gym3.Env,
    env_name: ENV_NAMES_T,
    policy: Callable[[np.ndarray], np.ndarray],
    num_trajs: int,
    n_actions: int = -1,
) -> List[Trajectory]:
    out: List[Trajectory] = []
    for _ in range(num_trajs):
        obs, reward, first = env.observe()
        info = env.get_info()[0]
        start_state = State(
            grid=info["grid"], agent_pos=info["agent_pos"], exit_pos=info["exit_pos"]
        )
        actions: List[np.ndarray] = []
        first = False
        while not first and (actions == -1 or len(actions) <= n_actions):
            action = policy(obs)
            env.act(action)
            obs, reward, first = env.observe()
            if not first:
                actions.append(action)
        out.append(Trajectory(start_state, np.stack(actions), env_name))
    return out


def collect_feature_trajs(
    env: gym3.Env,
    env_name: FEATURE_ENV_NAMES,
    policy: Callable[[np.ndarray], np.ndarray],
    num_trajs: int,
    n_actions: int = -1,
) -> List[FeatureTrajectory]:
    root_env = get_root_env(env)
    assert isinstance(root_env, FeatureEnv)
    out: List[FeatureTrajectory] = []
    for _ in range(num_trajs):
        obs, reward, first = env.observe()
        info = env.get_info()[0]
        start_state = State(
            grid=info["grid"], agent_pos=info["agent_pos"], exit_pos=info["exit_pos"]
        )
        actions: List[np.ndarray] = []
        features = [root_env.make_features()[0]]
        first = False
        while not first and (n_actions == -1 or len(actions) <= n_actions):
            action = policy(obs)
            env.act(action)
            obs, reward, first = env.observe()
            if not first:
                actions.append(action)
                features.append(root_env.make_features()[0])
        out.append(
            FeatureTrajectory(
                start_state, np.stack(actions), env_name, np.stack(features)
            )
        )
    return out


def compute_diffs(
    questions: List[Tuple[FeatureTrajectory, FeatureTrajectory]]
) -> np.ndarray:
    return np.stack(
        [
            question[0].features.sum(axis=0) - question[1].features.sum(axis=0)
            for question in questions
        ]
    )


def collect_feature_questions(
    env: gym3.Env,
    env_name: FEATURE_ENV_NAMES,
    policy: Callable[[np.ndarray], np.ndarray],
    n_questions: int,
    batch_size: int,
    n_actions: int = -1,
) -> List[Tuple[FeatureTrajectory, FeatureTrajectory]]:
    questions: List[Tuple[FeatureTrajectory, FeatureTrajectory]] = []

    while len(questions) < n_questions:
        trajs = collect_feature_trajs(env, env_name, policy, batch_size, n_actions)
        questions.extend(zip(trajs[::2], trajs[1::2]))
        diffs = compute_diffs(questions)
        diffs, indices = remove_zeros(diffs)
        diffs, tmp_indices = remove_duplicates(diffs)
        indices = indices[tmp_indices]
        questions = [questions[i] for i in indices]
    return questions
