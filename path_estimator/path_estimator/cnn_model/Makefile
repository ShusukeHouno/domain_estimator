GPU_NUM = 0
PORT_NUM = 8888
TENSOR_PORT = 6006

CONTAINER_NAME = tensor-keras

# DATASET_PATH = /data/aisl/uzawa/tmp
DATASET_PATH = /media/aisl/HD-PGF-A/Share_data/M2_Data_Share/create_dataset
# DATASET_PATH = /home/aisl/rs_dataset
RAW_DATASET_PATH = /home/aisl/rs_dataset
GENERATED_DATASET_PATH = /home/aisl/generated_dataset

docker-build:
	docker build -t tensor-keras .

docker-bash:
	docker run --rm -it --gpus all \
		-u $(id -u):$(id -g) \
		-p $(PORT_NUM):$(PORT_NUM) -p $(TENSOR_PORT):$(TENSOR_PORT) \
		-v $(shell pwd):/tf \
		-w /tf tensor-keras

docker-bash-connect-dataset:
	docker run --rm -it --gpus $(GPU_NUM) \
		-u $(shell id -u):$(shell id -g) \
		-p $(PORT_NUM):$(PORT_NUM) -p 8080:$(TENSOR_PORT) tensorboard \
		-v $(shell pwd):/tf -v $(DATASET_PATH):/create_dataset \
		-w /tf tensor-keras

docker-for-training:
	docker run --rm -it --gpus all \
		-u $(shell id -u):$(shell id -g) \
		-p $(PORT_NUM):$(PORT_NUM) -p $(TENSOR_PORT):$(TENSOR_PORT) \
		-v $(shell pwd):/tf \
		-v $(RAW_DATASET_PATH):/raw_dataset \
		-v $(GENERATED_DATASET_PATH):/generated_dataset \
		-v ~/.Xauthority:/root/.Xauthority \
		-e DISPLAY=$(hostname):0 \
		-w /tf tensor-keras

tensor-board:
	tensorboard --logdir $(shell pwd)/tensorboard/log/ --bind_all


dataset-generate:
	python3 tool/path_dataset/csv_generate.py /raw_dataset/sakaki_trial2_trajectory.csv \
	-d -p /dataset -s /sakaki_trial1