# Answering Machine

A web-app around a [Q&A BERT model](https://tfhub.dev/see--/bert-uncased-tf2-qa/1). This app is mostly written
to practice with [TF Serving](https://www.tensorflow.org/tfx/guide/serving) and NLP models.

## Installation

To install the web-app locally it is recommended to use a virtual environment:

``` sh
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the model service, docker is required.

## Run

The TF serving container is pulled from the [docker hub](https://hub.docker.com/r/tensorflow/serving) and is run with
the following command:

``` sh
docker run -p 8501:8501 \
	--mount type=bind,source=/Users/timdeklijn/projects/hackathon-model-zoo/model/,target=/models/bert_qa \
	-e MODEL_NAME=bert_qa \
	-t tensorflow/serving
``` 

Or simply run:

``` sh
make docker_run
```

The web-app that communicates with the container through API calls is started with:

``` sh
python app.py
```

## TODO

- [x] Make command to download the model
- [x] Move downloaded model to correct folder
- [ ] Make mounting of model not hardcoded
- [ ] Containerize the web-app
  - [ ] Create a docker-compose file to spin up the whole app
- [ ] Scrape a website instead of copy/pasting the text
- [ ] Shorten text by removing stop words - use something like `spacy`