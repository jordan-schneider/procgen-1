import sqlite3

from experiment_server.app import DATABASE
from experiment_server.query import get_random_question
from experiment_server.serialize import serialize


def test_serialize_question():
    conn = sqlite3.connect(DATABASE)
    question = get_random_question(conn, question_type="traj", env="miner", length=10)
    serialize(question)
