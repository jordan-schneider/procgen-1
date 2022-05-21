import flask
from flask import Flask, jsonify, render_template, request, url_for

app = Flask(__name__)


@app.route("/")
def main():
    url_for("static", filename="script.js")
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def ingest():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    print(request.get_json()["message"])
    return "OK"
