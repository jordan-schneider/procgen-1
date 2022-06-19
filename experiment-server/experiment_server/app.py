import sqlite3
from logging.config import dictConfig

import arrow
from flask import Flask, g, jsonify, render_template, request

from experiment_server.query import (
    get_random_question,
    insert_answer,
    insert_question,
    insert_traj,
)
from experiment_server.SECRETS import SECRET_KEY
from experiment_server.serialize import serialize
from experiment_server.type import Answer, Trajectory, assure_modality

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
DATABASE = (
    "/home/joschnei/web-procgen/experiment-server/experiment_server/experiments.db"
)
app.secret_key = SECRET_KEY


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
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

    id = insert_answer(
        get_db(),
        Answer(
            user_id=0,
            question_id=json["id"],  # type: ignore
            answer=json["answer"] == "right",  # type: ignore
            start_time=arrow.get(json["startTime"]).isoformat(),  # type: ignore
            end_time=arrow.get(json["stopTime"]).isoformat(),  # type: ignore
        ),
    )

    return jsonify({"success": True, "answer_id": id})


@app.route("/random_question", methods=["POST"])
def request_random_question():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    spec = request.get_json()

    env = spec["env"]  # type: ignore
    length = spec["lengths"][0]  # type: ignore
    modality = spec["types"][0]  # type: ignore
    exclude_ids = spec["exclude_ids"]  # type: ignore

    try:
        modality = assure_modality(modality)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    conn = get_db()

    question = get_random_question(
        conn, question_type=modality, env=env, length=length, exclude_ids=exclude_ids
    )
    return serialize(question)


@app.route("/submit_question", methods=["POST"])
def submit_question():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    json = request.get_json()
    traj_ids: Tuple[int, int] = json["traj_ids"]  # type: ignore
    label = json["name"]  # type: ignore

    id = insert_question(
        conn=get_db(), traj_ids=traj_ids, algo="manual", env_name="miner", label=label
    )
    return jsonify({"success": True, "question_id": id})


@app.route("/submit_trajectory", methods=["POST"])
def submit_trajectory():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    json = request.get_json()

    id = insert_traj(
        get_db(),
        Trajectory(
            start_state=json["start_state"],  # type: ignore
            actions=json["actions"],  # type: ignore
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
