import logging
import os
import sqlite3
from logging.config import dictConfig
from pathlib import Path
from typing import Tuple

import arrow
import boto3  # type: ignore
import numpy as np
from flask import Flask, g, jsonify, render_template, request

from experiment_server.query import (
    get_named_question,
    get_random_question,
    insert_answer,
    insert_question,
    insert_traj,
)
from experiment_server.serialize import serialize
from experiment_server.type import Answer, State, Trajectory, assure_modality

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["wsgi"]},
    }
)

app = Flask(__name__, static_url_path="/assets")
DATABASE_PATH = os.environ["DATABASE_PATH"]
app.secret_key = os.environ["SECRET_KEY"]


s3 = boto3.client("s3")


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        logging.debug(f"Local database expected at {DATABASE_PATH}")
        if not Path(DATABASE_PATH).exists():
            logging.info("Downloading database from Amazon S3")
            s3.download_file("mrl-experiment-sqlite", "experiments.db", DATABASE_PATH)
        db = g._database = sqlite3.connect(DATABASE_PATH)
    return db


@app.route("/")
def index():
    return render_template("replay.html")


@app.route("/interact")
def interact():
    return render_template("interact.html")


@app.route("/record")
def record():
    return render_template("record.html")


@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    json = request.get_json()
    assert json is not None

    id = insert_answer(
        get_db(),
        Answer(
            user_id=0,
            question_id=json["id"],
            answer=json["answer"] == "right",
            start_time=arrow.get(json["startTime"]).isoformat(),
            end_time=arrow.get(json["stopTime"]).isoformat(),
        ),
    )

    return jsonify({"success": True, "answer_id": id})


@app.route("/random_question", methods=["POST"])
def request_random_question():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    spec = request.get_json()
    assert spec is not None

    env = spec["env"]
    length = spec["lengths"][0]
    modality = spec["types"][0]
    exclude_ids = spec["exclude_ids"]

    try:
        modality = assure_modality(modality)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    conn = get_db()

    question = get_random_question(
        conn, question_type=modality, env=env, length=length, exclude_ids=exclude_ids
    )
    return serialize(question)


@app.route("/named_question", methods=["POST"])
def request_named_question():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    spec = request.get_json()
    assert spec is not None

    conn = get_db()
    question = get_named_question(conn=conn, name=spec["name"])
    return serialize(question)


@app.route("/submit_question", methods=["POST"])
def submit_question():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    json = request.get_json()
    assert json is not None
    traj_ids: Tuple[int, int] = json["traj_ids"]
    label = json["name"]

    id = insert_question(
        conn=get_db(), traj_ids=traj_ids, algo="manual", env_name="miner", label=label
    )
    return jsonify({"success": True, "question_id": id})


@app.route("/submit_trajectory", methods=["POST"])
def submit_trajectory():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    json = request.get_json()
    assert json is not None

    json_state = json["start_state"]

    id = insert_traj(
        get_db(),
        Trajectory(
            start_state=State.from_json(json_state),
            actions=np.array(json["actions"]),
            env_name="miner",
            modality="traj",
        ),
    )

    return jsonify({"success": True, "trajectory_id": id})


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()
    if Path(DATABASE_PATH).exists():
        logging.info("Uploading database to Amazon S3")
        # s3.upload_file(DATABASE_PATH, "mrl-experiment-sqlite", "experiments.db")
        pass
