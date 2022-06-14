import sqlite3
from logging.config import dictConfig

import arrow
from flask import (
    Flask,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from experiment_server.query import (
    create_user,
    get_random_question,
    insert_answer,
    insert_question,
    insert_traj,
)
from experiment_server.SECRETS import SECRET_KEY
from experiment_server.serialize import serialize
from experiment_server.types import (
    Answer,
    Demographics,
    Question,
    Trajectory,
    assure_modality,
)

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


@app.route("/submit", methods=["POST"])
def submit():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    json = request.get_json()

    insert_answer(
        get_db(),
        Answer(
            user_id=0,
            question_id=json["id"],  # type: ignore
            answer=json["answer"] == "right",  # type: ignore
            start_time=arrow.get(json["startTime"]).isoformat(),  # type: ignore
            end_time=arrow.get(json["stopTime"]).isoformat(),  # type: ignore
        ),
    )

    return jsonify({"success": True})


@app.route("/question", methods=["POST"])
def request_question():
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

    insert_question(get_db(), traj_ids, "manual", "miner")


@app.route("/login", methods=["POST"])
def login():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    # TODO: Maybe this is unnecessary because we're not actually using ExternalQuestion
    # TODO: Make sure this redirect works on preview. Consider just calling the function directly.
    if request.args.get("assignmentId") == "ASSIGNMENT_ID_NOT_AVAILABLE":
        return redirect(url_for("index"))

    session["assignmentId"] = request.args.get("assignmentId")
    session["hitId"] = request.args.get("hitId")
    session["turkSubmitTo"] = request.args.get("turkSubmitTo")
    session["workerId"] = request.args.get("workerId")

    # TODO: Get demographics from mturk somehow
    create_user(get_db(), session["workerId"], Demographics(age=-1))
    return redirect(url_for("index"))


@app.route("/record", methods=["POST"])
def record():
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


@app.route("/logout")
def logout():
    session.pop("assignmentId", None)
    session.pop("hitId", None)
    session.pop("turkSubmitTo", None)
    session.pop("workerId", None)

    # TODO: Whatever mturk wants me to return


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()
