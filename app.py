import os

from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField

from answering_machine import do_inference
import inputs

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret Key for forms


class InputForm(FlaskForm):
    """
    Flask-WTF form used to create an input form on the index page.

    The defaults are loaded from inputs.py
    """
    question = TextAreaField("Ask your question", default=inputs.question)
    text = TextAreaField("Supply your text", default=inputs.text)


@app.route("/")
def home():
    """
    Index page. Renders the index.html template which requires an input form

    Returns
    -------
    Renders the index.html page.
    """
    form = InputForm()
    return render_template("index.html", form=form)


@app.route("/answer", methods=["POST"])
def answer():
    """
    Process the web-form, send input to the answering machine, display the output
    by rendering the answer.html page.

    Returns
    -------
    Renders the output.html page which containst the input text, input question and
    the answer from the model.
    """
    # Process form
    text = request.form["text"]
    question = request.form["question"]
    # Send input to model and retreive the answer
    answer = do_inference(question, text)
    # Render page
    return render_template("answer.html", text=text, question=question, answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
