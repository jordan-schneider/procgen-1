import sqlite3
from pathlib import Path
from typing import List

import arrow
import fire  # type: ignore
import pandas as pd  # type: ignore
import remote_sqlite  # type: ignore


def main(db_path: Path) -> None:
    db = remote_sqlite.RemoteSqlite(db_path)
    db.pull(db.fspath)
    times = make_times(db.con)
    min_time = times.groupby("user_id").duration.min()
    good_users = min_time[min_time > pd.Timedelta("10 seconds")].index
    mturk_codes = get_codes_from_users(db.con, good_users)
    print(mturk_codes)


def make_times(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a dataframe of all times in the database by user."""
    cursor = conn.execute(
        "SELECT user_id, question_id, start_time, end_time FROM answers"
    )

    data = [
        (user_id, question_id, arrow.get(start_time), arrow.get(end_time))
        for user_id, question_id, start_time, end_time in cursor
    ]

    df = pd.DataFrame(
        data, columns=["user_id", "question_id", "start_time", "end_time"]
    )
    df["duration"] = df["end_time"] - df["start_time"]

    return df


def get_codes_from_users(conn: sqlite3.Connection, users: pd.Index) -> List[str]:
    """Return a dataframe of all MTurk codes for the given users."""
    users_str = ",".join(str(user) for user in users)
    cursor = conn.execute(f"SELECT payment_code FROM users WHERE id IN ({users_str})")

    return [code for (code,) in cursor]


if __name__ == "__main__":
    fire.Fire(main)
