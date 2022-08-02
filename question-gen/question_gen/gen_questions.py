import pickle as pkl
import sqlite3
from itertools import tee
from msilib.schema import Feature
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

import fire  # type: ignore
import numpy as np
from experiment_server.query import insert_question, insert_traj, save_questions
from experiment_server.type import DataModality, FeatureTrajectory, Trajectory
from linear_procgen import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen import make_env
from procgen.env import ENV_NAMES_T, ProcgenGym3Env

from question_gen.biyik import successive_elimination
from question_gen.gen_trajectory import (
    collect_feature_questions,
    collect_trajs,
    compute_diffs,
)
from question_gen.random_policy import RandomGridPolicy, RandomPolicy
from question_gen.util import setup_logging

Question = Tuple[Trajectory, Trajectory]
FeatureQuestion = Tuple[FeatureTrajectory, FeatureTrajectory]


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
        env=env,
        env_name=env_name,
        modality=question_type,
        policy=RandomGridPolicy(env, rng),
        n_questions=init_questions,
        batch_size=num_trajs,
        n_actions=n_actions,
    )
    reward_samples = centered_reward_samples(
        reward, reward_variance, num_reward_samples, rng
    )
    questions = select_infogain_questions(
        questions, reward_samples, num_questions, init_questions
    )
    save_questions(conn, questions, "infogain", env_name)
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


def build_questions_from_trajs(
    traj_paths: Sequence[Path],
    n_by_policy: int,
    n_by_done: int,
    rng: np.random.Generator,
) -> List[FeatureQuestion]:
    questions: List[FeatureQuestion] = []
    for question_block in build_questions_by_policy(traj_paths, n_by_policy).values():
        questions.extend(question_block)
    for question_block in build_questions_by_done(traj_paths, n_by_done, rng).values():
        questions.extend(question_block)
    return questions


def build_questions_by_policy(
    traj_paths: Sequence[Path], n: int
) -> Dict[Tuple[Path, Path], List[Tuple[FeatureTrajectory, FeatureTrajectory]]]:
    policies = collect_policies(traj_paths)
    n_policies = len(policies)
    n_pairs = int(n_policies * (n_policies - 1) / 2)
    questions_per_pair = n // n_pairs
    trajs_per_policy = collect_even_trajs_per_policy(
        traj_paths, policies, questions_per_pair
    )
    return {
        (first, second): list(zip(trajs_per_policy[first], trajs_per_policy[second]))
        for first, second in pairwise(trajs_per_policy.keys())
    }


def build_questions_by_done(
    traj_paths: Sequence[Path], n: int, rng: np.random.Generator
) -> Dict[Tuple[bool, bool], List[Tuple[FeatureTrajectory, FeatureTrajectory]]]:
    done_trajs, not_done_trajs = collect_even_trajs_per_done(traj_paths, n // 3)
    return {
        (True, True): list(zip(done_trajs, rng.permutation(done_trajs))),
        (True, False): list(zip(done_trajs, not_done_trajs)),
        (False, False): list(zip(not_done_trajs, rng.permutation(not_done_trajs))),
    }


def collect_even_trajs_per_policy(
    traj_paths: Sequence[Path], policies: Set[Path], n: int
) -> Dict[Path, List[FeatureTrajectory]]:
    done_paths = 0
    out: Dict[Path, List[FeatureTrajectory]] = {policy: [] for policy in policies}
    while done_paths < len(traj_paths):
        for path in traj_paths:
            data = pkl.load(path.open("rb"))
            for policy in policies:
                rows_needed = n - len(out[policy])
                if rows_needed == 0:
                    continue
                new_rows = data.df[data.df.policy == policy]
                new_rows = new_rows.iloc[: min(rows_needed, len(new_rows))]

                out[policy].extend(
                    FeatureTrajectory(
                        start_state=row.grid[0],
                        actions=row.actions,
                        env_name=row.env,
                        modality="traj",
                        features=row.features,
                        reason=f"From policy {policy}",
                    )
                    for row in new_rows
                )
                if len(out[policy]) == n:
                    done_paths += 1
    return out


def collect_even_trajs_per_done(
    traj_paths: Sequence[Path], n: int
) -> Tuple[List[FeatureTrajectory], List[FeatureTrajectory]]:
    done_trajs: List[FeatureTrajectory] = []
    not_done_trajs: List[FeatureTrajectory] = []
    while len(done_trajs) < n and len(not_done_trajs) < n:
        for path in traj_paths:
            data = pkl.load(path.open("rb"))

            done_rows_needed = n - len(done_trajs)
            if done_rows_needed > 0:
                new_rows = data.df[data.df.features[6] > 0]
                new_rows = new_rows.iloc[: min(done_rows_needed, len(new_rows))]
                done_trajs.extend(
                    FeatureTrajectory(
                        start_state=row.grid[0],
                        actions=row.actions,
                        env_name=row.env,
                        modality="traj",
                        features=row.features,
                        reason="Trajectory finished",
                    )
                    for row in new_rows
                )

            not_done_rows_needed = n - len(not_done_trajs)
            if not_done_rows_needed > 0:
                new_rows = data.df[data.df.features[6] == 0]
                new_rows = new_rows.iloc[: min(not_done_rows_needed, len(new_rows))]
                not_done_trajs.extend(
                    FeatureTrajectory(
                        start_state=row.grid[0],
                        actions=row.actions,
                        env_name=row.env,
                        modality="traj",
                        features=row.features,
                        reason="Trajectory not finished",
                    )
                    for row in new_rows
                )
    return done_trajs, not_done_trajs


def collect_policies(traj_paths: Sequence[Path]) -> Set[Path]:
    policies = set()
    for path in traj_paths:
        data = pkl.load(path.open("rb"))
        policies.update(data.df.policies.unique())
    return policies


def select_infogain_questions(
    questions: List[FeatureQuestion],
    reward_samples: np.ndarray,
    n: int,
    init_questions: int = 200,
) -> List[FeatureQuestion]:
    question_diffs = compute_diffs([q for q, _ in questions])
    indices = successive_elimination(question_diffs, reward_samples, n, init_questions)
    return np.array(questions)[indices].tolist()


def gen_infogain_questions_from_saved_trajs(
    db_path: Path,
    traj_paths: List[Path],
    reward_path: Path,
    n_by_policy: int,
    n_by_done: int,
    n_questions: int,
    seed: int,
    reward_variance: float = 0.01,
    n_reward_samples: int = 1000,
    n_init_questions: int = 200,
):
    conn = sqlite3.connect(db_path)
    rng = np.random.default_rng(seed)
    reward = np.load(reward_path)
    reward_samples = centered_reward_samples(
        reward, reward_variance, n_reward_samples, rng
    )

    questions = build_questions_from_trajs(traj_paths, n_by_policy, n_by_done, rng)
    questions = select_infogain_questions(
        questions, reward_samples, n_questions, n_init_questions
    )

    save_questions(conn, questions, "infogain", env_name=questions[0][0].env_name)

    conn.close()


def centered_reward_samples(
    reward: np.ndarray, variance: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.multivariate_normal(
        mean=reward,
        cov=np.eye(reward.shape[0]) * variance,
        size=n,
    )


def clean_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    remove_duplicate_questions(conn)
    remove_orphan_trajectories(conn)

    conn.commit()
    conn.close()


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


if __name__ == "__main__":
    fire.Fire(
        {
            "gen_random_questions": gen_random_questions,
            "gen_infogain_questions": gen_batch_infogain_questions,
            "infogain_from_saved_trajs": gen_infogain_questions_from_saved_trajs,
            "init_db": init_db,
            "clean_db": clean_db,
        }
    )
