import flask
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_url_path="/assets")


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def ingest():
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    print(request.get_json()["message"])
    return "OK"
