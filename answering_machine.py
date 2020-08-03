import json

import numpy as np
import requests
from transformers import BertTokenizer

# MODEL API URL
TF_SERVING_URL = "http://localhost:8501/v1/models/bert_qa:predict"

# Tokenizer used in pre-processing.
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)


def prep_inputs(question, text):
    """
    Convert text and question input to tokens, prep API payload.

    Create model inputs from input question ad text by converting the text to
    tokens using the tokenizer. Then create a token list, a list of word ids,
    the input mask and the input type ids.

    Parameters
    ----------
    question: str
        input question
    text: str
        input text

    Returns
    -------
    tokens: List[int/str]
        list of tokens separated by special characters
    payload: Dict
        dict formatted so to be sent to the model API
    """
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
    """
    Translate the API response model output to plain text again

    Parameters
    ----------
    answer: Dict[List[int]]
        Model output containing metadata and some token vectors that can
        be translated back to plain text.
    tokens: List[int]
        Tokens from text and input question

    Returns
    -------
    answer: str
        Plain text answer to the question
    """
    # Extract needed data from API response
    short_start_nums = answer["outputs"]["tf_bert_for_natural_question_answering"][0][1:]
    short_end_nums = answer["outputs"]["tf_bert_for_natural_question_answering_1"][0][1:]

    # Process Model output
    short_start = np.argmax(short_start_nums) + 1
    short_end = np.argmax(short_end_nums) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    # Convert answer back to text
    return tokenizer.convert_tokens_to_string(answer_tokens)


def do_inference(question, text):
    """
    Main method for the answering machine.

    Pass the question and text through the pre-processor. Send the
    tokenized question/text to the tensorflow serving API and post-process
    the response.

    Parameters
    ----------
    question: str
        Input question obout the text
    text: str
        Input text, should not be more then 512 words long

    Returns
    -------
    answer: str
        Answer to the question about the text
    """
    # pre-process inputs
    tokens, payload = prep_inputs(question, text)
    # Send inputs to model
    r = requests.post(TF_SERVING_URL, data=json.dumps(payload))
    # Translate model output to plain text
    return translate_answer(r.json(), tokens)
