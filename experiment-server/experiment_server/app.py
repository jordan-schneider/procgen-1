import sqlite3
from typing import cast

from flask import Flask, g, jsonify, render_template, request

from experiment_server.query import get_random_question
from experiment_server.serialize import serialize
from experiment_server.types import DataModality

app = Flask(__name__, static_url_path="/assets")
DATABASE = (
    "/home/joschnei/web-procgen/experiment-server/experiment_server/experiments.db"
)


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def ingest():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    json = request.get_json()
    if json is None:
        return jsonify({"error": "Invalid JSON"}), 400
    print(json["message"])
    return "OK"


@app.route("/question", methods=["GET"])
def question():
    if request.method != "GET":
        return jsonify({"error": "Method not allowed"}), 405
    env = request.args.get("env")
    length_s = request.args.get("left_length")
    modality = request.args.get("left_type")
    if not (modality == "state" or modality == "action" or modality == "traj"):
        return jsonify({"error": "Invalid modality"}), 400
    modality = cast(DataModality, modality)
    if env is None:
        return jsonify({"error": "Missing env"}), 400
    if length_s is None:
        return jsonify({"error": "Missing length"}), 400
    length = int(length_s)

    conn = get_db()

    question = get_random_question(conn, question_type=modality, env=env, length=length)
    return serialize(question)


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()
