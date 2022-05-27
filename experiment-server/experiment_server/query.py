import pickle
import sqlite3

from experiment_server.types import Answer, DataModality, Question, Trajectory


def get_random_question(
    conn: sqlite3.Connection, question_type: DataModality, env: str, length: int
) -> Question:
    cursor = conn.execute(
        """SELECT
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
        questions as q
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
        ORDER BY RANDOM() LIMIT 1;
        """,
        {"question_type": question_type, "length": length, "env": env},
    )
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


def submit_answer(conn: sqlite3.Connection, answer: Answer):
    conn.execute(
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
    conn.commit()
