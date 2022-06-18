import sqlite3
from pathlib import Path
from typing import Literal

import fire  # type: ignore
import numpy as np
from linear_procgen import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen import make_env
from procgen.env import ENV_NAMES_T, ProcgenGym3Env

from experiment_server.biyik import successive_elimination
from experiment_server.gen_trajectory import (
    collect_feature_questions,
    collect_trajs,
    compute_diffs,
)
from experiment_server.query import insert_question, insert_traj
from experiment_server.random_policy import RandomGridPolicy, RandomPolicy
from experiment_server.type import DataModality
from experiment_server.util import setup_logging


def init_db(db_path: Path, schema_path: Path):
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r") as f:
        schema = f.read()
    conn.executescript(schema)
    conn.commit()
    conn.close()


def correct_n_actions(question_type: DataModality, n_actions: int) -> int:
    if question_type == "state":
        return 0
    elif question_type == "action":
        return 1
    elif question_type == "traj":
        return n_actions
    else:
        raise ValueError(f"Question type {question_type} not recognized")


def gen_random_questions(
    db_path: Path,
    env: ENV_NAMES_T,
    num_questions: int,
    question_type: DataModality,
    n_actions: int = 10,
    seed: int = 0,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(verbosity)
    conn = sqlite3.connect(db_path)
    env_name = env

    n_actions = correct_n_actions(question_type, n_actions)

    env = ProcgenGym3Env(1, env)
    questions = 0
    while questions < num_questions:
        trajs = collect_trajs(
            env,
            env_name,
            policy=RandomPolicy(env, np.random.default_rng(seed)),
            num_trajs=num_questions * 2,
            modality=question_type,
            n_actions=n_actions,
        )

        for traj_1, traj_2 in zip(trajs[::2], trajs[1::2]):
            if traj_1 == traj_2:
                continue
            traj_1_id = insert_traj(conn, traj_1)
            traj_2_id = insert_traj(conn, traj_2)
            insert_question(conn, (traj_1_id, traj_2_id), "random", env_name)
            questions += 1
    conn.commit()
    conn.close()


def gen_batch_infogain_questions(
    db_path: Path,
    true_reward_path: Path,
    env: FEATURE_ENV_NAMES,
    num_questions: int,
    question_type: DataModality,
    num_reward_samples: int,
    init_questions: int,
    num_trajs: int,
    reward_variance: float = 0.01,
    n_actions: int = 10,
    seed: int = 0,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(verbosity)
    conn = sqlite3.connect(db_path)
    reward = np.load(true_reward_path)
    env_name = env

    n_actions = correct_n_actions(question_type, n_actions)

    rng = np.random.default_rng(seed)

    env = make_env(env, num=1, reward=1)

    questions = collect_feature_questions(
        env,
        env_name,
        RandomGridPolicy(env, rng),
        init_questions,
        num_trajs,
        question_type,
        n_actions,
    )
    question_diffs = compute_diffs(questions)

    reward_samples = rng.multivariate_normal(
        mean=reward,
        cov=np.eye(question_diffs.shape[1]) * reward_variance,
        size=num_reward_samples,
    )

    indices = successive_elimination(
        question_diffs, reward_samples, num_questions, init_questions
    )
    # np array over questions for scalar list index
    for traj_1, traj_2 in np.array(questions)[indices]:
        traj_1_id = insert_traj(conn, traj_1)
        traj_2_id = insert_traj(conn, traj_2)
        insert_question(conn, (traj_1_id, traj_2_id), "infogain", env_name)
    conn.commit()
    conn.close()


def remove_duplicate_questions(conn: sqlite3.Connection) -> None:
    conn.execute(
        """DELETE FROM questions WHERE id in (
            SELECT id FROM (
                SELECT questions.id,
                       t1.start_state as start1, t1.actions as actions1, t1.env as env1, t1.modality as modality1,
                       t2.start_state as start2, t2.actions as actions2, t2.env as env2, t2.modality as modality2
                FROM questions
                INNER JOIN trajectories t1 ON questions.first_id = t1.id
                INNER JOIN trajectories t2 ON questions.second_id = t2.id
            ) WHERE start1 = start2 AND actions1 = actions2 AND env1 = env2 AND modality1 = modality2
        );
        """
    )
    conn.commit()


def remove_orphan_trajectories(conn: sqlite3.Connection) -> None:
    conn.execute(
        """DELETE FROM trajectories WHERE id NOT IN (
            SELECT first_id AS id FROM questions UNION ALL
            SELECT second_id AS id FROM questions);
        """
    )
    conn.commit()


def clean_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    remove_duplicate_questions(conn)
    remove_orphan_trajectories(conn)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    fire.Fire(
        {
            "gen_random_questions": gen_random_questions,
            "gen_infogain_questions": gen_batch_infogain_questions,
            "init_db": init_db,
            "clean_db": clean_db,
        }
    )
