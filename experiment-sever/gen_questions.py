import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Literal, Tuple

import fire
import gym3
import numpy as np

# TODO: Move linear feature envs from mrl to separate linear_procgen package.
from linear_procgen import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen import make_env
from numpy.random import Generator
from procgen.env import ENV_NAMES, ProcgenGym3Env

from biyik import successive_elimination


def init_db(db_path: Path, schema_path: Path):
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close()


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


def insert_traj(conn: sqlite3.Connection, traj: Trajectory) -> int:
    c = conn.execute(
        "INSERT INTO trajectories (start_state, actions, env, modality) VALUES (:start_state, :actions, :env, :modality)",
        {
            "start_state": pickle.dumps(traj.start_state),
            "actions": pickle.dumps(traj.actions),
            "env": traj.env_name,
            "modality": "traj",
        },
    )
    return int(c.lastrowid)


def insert_question(
    conn: sqlite3.Connection,
    traj_ids: Tuple[int, int],
    algo: Literal["random", "infogain"],
) -> int:
    c = conn.execute(
        "INSERT INTO questions (first_id, second_id, modality, algorithm) VALUES (:first_id, :second_id, :modality, :algo)",
        {
            "first_id": traj_ids[0],
            "second_id": traj_ids[1],
            "modality": "traj",
            "algo": algo,
        },
    )
    return int(c.lastrowid)


class RandomPolicy:
    def __init__(self, env: gym3.Env, rng: Generator):
        self.actions = env.ac_space.eltype.n
        self.num = env.num
        self.rng = rng

    def __call__(self, ob):
        return self.rng.integers(low=0, high=self.actions, size=(self.num,))


def gen_random_questions(
    db_path: Path, env: ENV_NAMES, num_questions: int, seed: int = 0
) -> None:
    conn = sqlite3.connect(db_path)
    env_name = env

    env = ProcgenGym3Env(1, env)
    trajs = collect_trajs(
        env,
        env_name,
        policy=RandomPolicy(env, np.random.default_rng(seed)),
        num_trajs=num_questions * 2,
    )

    for traj_1, traj_2 in zip(trajs[::2], trajs[1::2]):
        traj_1_id = insert_traj(conn, traj_1)
        traj_2_id = insert_traj(conn, traj_2)
        insert_question(conn, (traj_1_id, traj_2_id), "random")
    conn.commit()
    conn.close()


def remove_redundant(vecs: np.ndarray):
    # TODO: Copy from vav/mrl project
    pass


def gen_batch_infogain_questions(
    db_path: Path,
    env: FEATURE_ENV_NAMES,
    num_reward_samples: int,
    num_questions: int,
    init_questions: int,
    num_trajs: int,
    seed: int = 0,
) -> None:
    conn = sqlite3.connect(db_path)
    env_name = env

    rng = np.random.default_rng(seed)

    env = make_env(env)
    trajs = collect_feature_trajs(
        env,
        env_name,
        policy=RandomPolicy(env, rng),
        num_trajs=num_trajs,
    )

    questions = list(zip(trajs[::2], trajs[1::2]))
    question_diffs = np.array(
        question[0].features - question[1].features for question in questions
    )
    question_diffs = remove_redundant(question_diffs)

    # TODO: Implement non-uniform priors
    reward_samples = rng.multivariate_normal(
        mean=np.zeros_like(question_diffs[0]),
        cov=np.eye(question_diffs.shape[0]),
        size=num_reward_samples,
    )

    indices = successive_elimination(
        question_diffs, reward_samples, num_questions, init_questions
    )

    for traj_1, traj_2 in questions[indices]:
        traj_1_id = insert_traj(conn, traj_1)
        traj_2_id = insert_traj(conn, traj_2)
        insert_question(conn, (traj_1_id, traj_2_id), "infogain")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    fire.Fire(
        {
            "gen_random_questions": gen_random_questions,
            "gen_infogain_questions": gen_batch_infogain_questions,
            "init_db": init_db,
        }
    )
