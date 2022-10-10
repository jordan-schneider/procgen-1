import logging
from copy import deepcopy
from typing import List, Tuple

import gym3  # type: ignore
import numpy as np
from experiment_server.type import FeatureTrajectory, State
from linear_procgen import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen.util import get_root_env


def gen_action_pairs(
    env: gym3.Env,
    env_name: FEATURE_ENV_NAMES,
    n_pairs: int,
    n_steps: int,
    rng: np.random.RandomState,
) -> List[Tuple[FeatureTrajectory, FeatureTrajectory]]:
    assert n_pairs <= n_steps
    # TODO: This is inefficient, we only need to branch every n_steps // n_pairs steps.
    pairs = get_branched_path(env, env_name, n_steps, n_steps, rng)
    logging.debug(f"Generated {len(pairs)} pairs")
    step_size = n_steps // n_pairs
    end = step_size * n_pairs
    logging.debug(f"Step size: {step_size}, end: {end}")
    return pairs[:end:step_size]


def get_branched_path(
    env: gym3.Env,
    env_name: FEATURE_ENV_NAMES,
    n_pairs: int,
    n_steps: int,
    rng: np.random.RandomState,
) -> List[Tuple[FeatureTrajectory, FeatureTrajectory]]:
    assert n_pairs <= n_steps
    out: List[Tuple[FeatureTrajectory, FeatureTrajectory]] = []

    root_env = get_root_env(env)
    for _ in range(n_pairs):
        start_cstate = root_env.get_state()
        start_state = build_state(env)
        start_feature = root_env.get_features()[0]

        first_action = gym3.types_np.sample(env.ac_space, bshape=(env.num,), rng=rng)

        second_action = first_action
        while np.array_equal(second_action, first_action):
            second_action = gym3.types_np.sample(
                env.ac_space, bshape=(env.num,), rng=rng
            )

        env.act(first_action)
        first_features = root_env.get_features()[0]

        root_env.set_state(start_cstate)
        assert start_state == build_state(env)
        env.act(second_action)
        second_features = root_env.get_features()[0]

        first_traj = FeatureTrajectory(
            start_state=start_state,
            actions=np.array([first_action]),
            env_name=env_name,
            modality="action",
            features=np.array([start_feature, first_features]),
        )
        second_traj = FeatureTrajectory(
            start_state=start_state,
            actions=np.array([first_action]),
            env_name=env_name,
            modality="action",
            features=np.array([start_feature, second_features]),
        )
        out.append((first_traj, second_traj))
    return out


def build_state(env: gym3.Env) -> State:
    info = env.get_info()[0]
    return State(
        grid=info["grid"],
        grid_shape=tuple(info["grid_size"]),
        agent_pos=tuple(info["agent_pos"]),
        exit_pos=tuple(info["exit_pos"]),
    )
