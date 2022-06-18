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
    left.start_state as left_start,
    left.actions as left_actions,
    left.length as left_length,
    left.modality as left_modality,
    right.start_state as right_start,
    right.actions as right_actions,
    right.length as right_length,
    right.modality as right_modality
FROM 
    (SELECT * FROM questions {f"WHERE questions.id NOT IN ({excl_list})" if len(excl_list) > 0 else ""}) as q
    LEFT JOIN trajectories as left ON
        q.first_id=left.id
    LEFT JOIN trajectories as right ON
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
        first_traj=Trajectory(
            pickle.loads(left_start), pickle.loads(left_actions), env, left_modality
        ),
        second_traj=Trajectory(
            pickle.loads(right_start), pickle.loads(right_actions), env, right_modality
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
) -> int:
    cursor = conn.execute(
        "INSERT INTO questions (first_id, second_id, algorithm, env) VALUES (:first_id, :second_id, :algo, :env)",
        {
            "first_id": traj_ids[0],
            "second_id": traj_ids[1],
            "algo": algo,
            "env": env_name,
        },
    )
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
