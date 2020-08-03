import os

from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField

from answering_machine import do_inference
import inputs

app = Flask(__name__)
app.secret_key = os.urandom(24)


class InputForm(FlaskForm):
    question = TextAreaField("Ask your question", default=inputs.question)
    text = TextAreaField("Supply your text", default=inputs.text)


@app.route("/")
def home():
    form = InputForm()
    return render_template("index.html", form=form)


@app.route("/answer", methods=["POST"])
def answer():
    text = request.form["text"]
    question = request.form["question"]
    answer = do_inference(question, text)
    return render_template("answer.html", text=text, question=question, answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
