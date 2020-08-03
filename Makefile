# Run a tensorflow/serving container, mount the model to /models/... and set the ports.
docker_run:
	echo "Run the tensorflow serving container and mount the model"
	docker run -p 8501:8501 \
	--mount type=bind,source=/Users/timdeklijn/projects/hackathon-model-zoo/model/,target=/models/bert_qa \
	-e MODEL_NAME=bert_qa \
	-t tensorflow/serving

download_model:
	curl https://tfhub.dev/see--/bert-uncased-tf2-qa/1 --output model/2/