import json

import inputs
import requests
import tensorflow as tf
from transformers import BertTokenizer

TF_SERVING_URL = "http://localhost:8501/v1/models/bert_qa:predict"

INPUT = inputs.question
TEXT = inputs.text

tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)


def prep_inputs(question, text):
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + paragraph_tokens + ["[SEP]"]
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (
            len(paragraph_tokens) + 1
    )
    return (
        tokens,
        {
            "inputs": {
                "input_word_ids": [input_word_ids],
                "input_mask": [input_mask],
                "input_type_ids": [input_type_ids],
            }
        },
    )


def translate_answer(answer, tokens):
    print(answer)
    short_start_nums = answer["outputs"]["tf_bert_for_natural_question_answering"][0][1:]
    short_end_nums = answer["outputs"]["tf_bert_for_natural_question_answering_1"][0][1:]

    short_start = tf.argmax(short_start_nums) + 1
    short_end = tf.argmax(short_end_nums) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    return tokenizer.convert_tokens_to_string(answer_tokens)


def do_inference(question, text):
    tokens, payload = prep_inputs(question, text)
    r = requests.post(TF_SERVING_URL, data=json.dumps(payload))
    return translate_answer(r.json(), tokens)
