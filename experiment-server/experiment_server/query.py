import logging
import pickle
import sqlite3
from typing import Optional, Sequence, Tuple

from experiment_server.type import (
    Answer,
    DataModality,
    Demographics,
    Question,
    QuestionAlgorithm,
    Trajectory,
)


def get_random_question(
    conn: sqlite3.Connection,
    question_type: DataModality,
    env: str,
    length: int,
    exclude_ids: Optional[Sequence[int]] = None,
) -> Question:
    if exclude_ids is None:
        exclude_ids = []
    excl_list = ", ".join(f":excl_{i}" for i in range(len(exclude_ids)))
    excl_values = {f"excl_{i}": id for i, id in enumerate(exclude_ids)}
    query_s = f"""
SELECT
    q.*,
    left.start_state AS left_start,
    left.actions AS left_actions,
    left.length AS left_length,
    left.modality AS left_modality,
    right.start_state AS right_start,
    right.actions AS right_actions,
    right.length AS right_length,
    right.modality AS right_modality
FROM 
    (SELECT * FROM questions {f"WHERE questions.id NOT IN ({excl_list})" if len(excl_list) > 0 else ""}) AS q
    LEFT JOIN trajectories AS left ON
        q.first_id=left.id
    LEFT JOIN trajectories AS right ON
        q.second_id=right.id
    WHERE
        left.length=:length
        AND right.length=:length
        AND left.modality=:question_type
        AND right.modality=:question_type
        AND q.env=:env
        AND left.env=:env
        AND right.env=:env 
    ORDER BY RANDOM() LIMIT 1;"""
    values = {
        "question_type": question_type,
        "length": length,
        "env": env,
        **excl_values,
    }
    logging.debug(f"Querying:\n{query_s}\nwith values:\n{values}")

    cursor = conn.execute(query_s, values)
    (
        id,
        first_id,
        second_id,
        algorithm,
        env,
        question_name,
        left_start,
        left_actions,
        left_length,
        left_modality,
        right_start,
        right_actions,
        right_length,
        right_modality,
    ) = next(cursor)
    # TODO: Swap pickle for dill
    return Question(
        id=id,
        trajs=(
            Trajectory(
                pickle.loads(left_start), pickle.loads(left_actions), env, left_modality
            ),
            Trajectory(
                pickle.loads(right_start),
                pickle.loads(right_actions),
                env,
                right_modality,
            ),
        ),
    )


def get_named_question(conn: sqlite3.Connection, name: str) -> Question:
    query_s = """
SELECT
    q.*,
    left.start_state AS left_start,
    left.actions AS left_actions,
    left.length AS left_length,
    left.modality AS left_modality,
    right.start_state AS right_start,
    right.actions AS right_actions,
    right.length AS right_length,
    right.modality AS right_modality
FROM 
    questions AS q
    LEFT JOIN trajectories AS left ON
        q.first_id=left.id
    LEFT JOIN trajectories AS right ON
        q.second_id=right.id
    WHERE
        q.label=:name
    ORDER BY RANDOM() LIMIT 1;"""
    values = {"name": name}
    logging.debug(f"Querying:\n{query_s}\nwith values:\n{values}")

    cursor = conn.execute(query_s, values)
    (
        id,
        first_id,
        second_id,
        algorithm,
        env,
        question_name,
        left_start,
        left_actions,
        left_length,
        left_modality,
        right_start,
        right_actions,
        right_length,
        right_modality,
    ) = next(cursor)
    # TODO: Swap pickle for dill
    return Question(
        id=id,
        trajs=(
            Trajectory(
                pickle.loads(left_start), pickle.loads(left_actions), env, left_modality
            ),
            Trajectory(
                pickle.loads(right_start),
                pickle.loads(right_actions),
                env,
                right_modality,
            ),
        ),
    )


def insert_answer(conn: sqlite3.Connection, answer: Answer) -> int:
    cursor = conn.execute(
        """
        INSERT INTO answers (user_id, question_id, answer, start_time, end_time) VALUES (:user_id, :question_id, :answer, :start_time, :end_time)
        """,
        {
            "user_id": answer.user_id,
            "question_id": answer.question_id,
            "answer": answer.answer,
            "start_time": answer.start_time,
            "end_time": answer.end_time,
        },
    )
    assert cursor.lastrowid is not None
    out = int(cursor.lastrowid)
    conn.commit()
    return out


def insert_traj(conn: sqlite3.Connection, traj: Trajectory) -> int:
    # TODO: Swap pickle for dill
    cursor = conn.execute(
        "INSERT INTO trajectories (start_state, actions, length, env, modality) VALUES (:start_state, :actions, :length, :env, :modality)",
        {
            "start_state": pickle.dumps(traj.start_state),
            "actions": pickle.dumps(traj.actions),
            "length": len(traj.actions) if traj.actions is not None else 0,
            "env": traj.env_name,
            "modality": traj.modality,
        },
    )
    assert cursor.lastrowid is not None
    out = int(cursor.lastrowid)
    conn.commit()
    return out


def insert_question(
    conn: sqlite3.Connection,
    traj_ids: Tuple[int, int],
    algo: QuestionAlgorithm,
    env_name: str,
    label: Optional[str] = None,
) -> int:
    label_schema = ", label" if label is not None else ""
    label_value = ", :label" if label is not None else ""
    query = f"INSERT INTO questions (first_id, second_id, algorithm, env{label_schema}) VALUES (:first_id, :second_id, :algo, :env{label_value})"
    values = {
        "first_id": traj_ids[0],
        "second_id": traj_ids[1],
        "algo": algo,
        "env": env_name,
    }
    if label is not None:
        values["label"] = label
    logging.debug(f"query={query}, values={values}")
    cursor = conn.execute(query, values)
    assert cursor.lastrowid is not None
    out = int(cursor.lastrowid)
    conn.commit()
    return out


def create_user(
    conn: sqlite3.Connection, user_id: int, demographics: Demographics
) -> None:
    conn.execute(
        """
        INSERT INTO users VALUES (:id, :sequence, :demographics)
        """,
        {
            "id": user_id,
            "sequence": 0,
            "demographics": pickle.dumps(demographics),
        },
    )
    conn.commit()
