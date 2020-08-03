# Tim de Klijn, 2020

# Download the model and save it in the folder 'model/1/'. This is important because we use the
# absolute path to mount the model to the tenserflow/serving container.
download_model:
	./download_model.sh

# Run a tensorflow/serving container, mount the model to /models/... and set the ports.
docker_run:
	@echo "Run the tensorflow serving container and mount the model"
	docker run -p 8501:8501 \
	--mount type=bind,source=/Users/timdeklijn/projects/hackathon-model-zoo/model/,target=/models/bert_qa \
	-e MODEL_NAME=bert_qa \
	--name tf-serve-bert-qa \
	-t tensorflow/serving

# Stop and remove the tensorflow/serving container
docker_stop:
	@echo "Stopping and removing docker container"
	docker stop tf-serve-bert-qa
	docker rm tf-serve-bert-qa
	@echo "Done"
