docker_run:
	docker run -p 8501:8501 \
	--mount type=bind,source=/Users/timdeklijn/projects/hackathon-model-zoo/backend/model/,target=/models/bert_qa \
	-e MODEL_NAME=bert_qa -t tensorflow/serving
