from dataclasses import dataclass
from typing import Callable, List, Tuple

import gym3
import numpy as np
from linear_procgen import ENV_NAMES as FEATURE_ENV_NAMES
from procgen.env import ENV_NAMES


@dataclass
class State:
    grid: np.ndarray
    agent_pos: Tuple[int, int]
    exit_pos: Tuple[int, int]


@dataclass
class Trajectory:
    start_state: State
    actions: List[int]
    env_name: str


@dataclass
class FeatureTrajectory(Trajectory):
    features: np.ndarray


def collect_trajs(
    env: gym3.Env,
    env_name: ENV_NAMES,
    policy: Callable[[np.ndarray], int],
    num_trajs: int,
) -> List[Trajectory]:
    out: List[Trajectory] = []
    for _ in range(num_trajs):
        obs, reward, first = env.observe()
        info = env.get_info()[0]
        start_state = State(
            grid=info["grid"], agent_pos=info["agent_pos"], exit_pos=info["exit_pos"]
        )
        actions = []
        while not first:
            action = policy(obs)
            env.act(action)
            obs, reward, first = env.observe()
            if not first:
                actions.append(action)
        out.append(Trajectory(start_state, actions, env_name))
    return out


def collect_feature_trajs(
    env: gym3.Env,
    env_name: FEATURE_ENV_NAMES,
    policy: Callable[[np.ndarray], int],
    num_trajs: int,
) -> List[FeatureTrajectory]:
    out: List[FeatureTrajectory] = []
    for _ in range(num_trajs):
        obs, reward, first = env.observe()
        info = env.get_info()[0]
        start_state = State(
            grid=info["grid"], agent_pos=info["agent_pos"], exit_pos=info["exit_pos"]
        )
        actions = []
        features = [info["features"]]
        while not first:
            action = policy(obs)
            env.act(action)
            obs, reward, first = env.observe()
            if not first:
                actions.append(action)
                info = env.get_info()[0]
                features.append(info["features"])
        out.append(
            FeatureTrajectory(start_state, actions, env_name, np.array(features))
        )
    return out
