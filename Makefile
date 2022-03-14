POD_NAME ?= dragline
STORAGE_BUCKET ?= fecambucket
STORAGE_IMAGE ?= docker.io/bitnami/minio:2021.4.6
STORAGE_CONTAINER_NAME ?= fecam-storage
STORAGE_ACCESS_KEY ?= minio-access-key
STORAGE_ACCESS_SECRET ?= minio-secret-key
STORAGE_PORT ?= 9000

IMAGE_NAMESPACE ?= localhost
APACHE_TIKA_IMAGE_NAME ?= apache-tika
APACHE_TIKA_IMAGE_TAG ?= latest
APACHE_TIKA_CONTAINER_NAME ?= apache-tika

DATA_DIR ?= "data"
ENV_NAME ?= $(shell conda env export --json | jq ".name")
DATA_FILE ?= "$(DATA_DIR)/wikipedia_20220220_pt.csv"
WIKIPEDIA_DATA_DIR ?= "$(DATA_DIR)/wikipedia"
WIKIPEDIA_DATA_FILES_COUNT ?= $(shell ls -l $(WIKIPEDIA_DATA_DIR) | wc -l)
WIKIPEDIA_DATASET_SIZE ?= $$(( $(WIKIPEDIA_DATA_FILES_COUNT) * 1000 ))
VOCAB_FILE ?= "$(DATA_DIR)/bertimbau_base_vocab.txt"
DATASET_SIZE ?= $(shell expr `cat $(DATA_FILE) | wc -l` - 1)
VOCAB_SIZE ?= $(shell cat $(VOCAB_FILE) | wc -l)
MODEL_PATH ?= "models/text_transformer_autoencoder"
SENTENCES_FILE ?= "$(DATA_FILE)/sentences_to_predict"
BATCH_SIZE ?= 32

python_script = PYTHONPATH=$(PWD) \
	TF_CPP_MIN_LOG_LEVEL=2 \
	DATA_FILE=$(DATA_FILE) \
	DATASET_SIZE=$(DATASET_SIZE) \
	WIKIPEDIA_DATA_DIR=$(WIKIPEDIA_DATA_DIR) \
	WIKIPEDIA_DATASET_SIZE=$(WIKIPEDIA_DATASET_SIZE) \
	WIKIPEDIA_DATA_FILES_COUNT=$(WIKIPEDIA_DATA_FILES_COUNT) \
	VOCAB_SIZE=$(VOCAB_SIZE) \
	VOCAB_FILE=$(VOCAB_FILE) \
	MODEL_PATH=$(MODEL_PATH) \
	SENTENCES_FILE=$(SENTENCES_FILE) \
	BATCH_SIZE=$(BATCH_SIZE) \
	python $(1)

.PHONY: download-models
download-models: python -m spacy download pt_core_news_lg
	python -m spacy download en_core_web_trf
	python -m spacy download en_core_web_lg

.PHONY: format
format:
	black gazettes/* scripts/*

.PHONY: find-aquisitions
find-aquisitions:
	python scripts/find-buy-relations.py


.PHONY: tests
tests:
	python -m unittest -f tests

.PHONY:destroy-pod
destroy-pod:
	podman pod rm --force --ignore $(POD_NAME)

.PHONY:create-pod
create-pod: destroy-pod
	podman pod create -p $(STORAGE_PORT):$(STORAGE_PORT) --name $(POD_NAME)

.PHONY: stop-storage
stop-storage:
	podman rm --force --ignore $(STORAGE_CONTAINER_NAME)

.PHONY: storage
storage: stop-storage start-storage

.PHONY:start-storage
start-storage:
	podman run -d --rm -ti \
	--name $(STORAGE_CONTAINER_NAME) \
	--pod $(POD_NAME) \
	-e MINIO_ACCESS_KEY=$(STORAGE_ACCESS_KEY) \
	-e MINIO_SECRET_KEY=$(STORAGE_ACCESS_SECRET) \
	-e MINIO_DEFAULT_BUCKETS=$(STORAGE_BUCKET):public \
	$(STORAGE_IMAGE)

.PHONY: download-files-sample
download-files-sample:
	python scripts/download_files.py

.PHONY: files-sample
files-sample:
	python scripts/sample_files.py

.PHONY: extract-text-from-sample-files
extract-text-from-sample-files:
	python scripts/extract_text_from_files.py

.PHONY: build-containers
build-containers:
	podman build --tag $(IMAGE_NAMESPACE)/$(APACHE_TIKA_IMAGE_NAME):$(APACHE_TIKA_IMAGE_TAG) -f resources/Dockerfile_apache_file resources

.PHONY: stop-apache-tika
stop-apache-tika:
	- podman rm --force $(APACHE_TIKA_CONTAINER_NAME)

.PHONY: start-apache-tika
start-apache-tika: stop-apache-tika
	podman run -d  --name $(APACHE_TIKA_CONTAINER_NAME) -p 9998:9998 \
		apache/tika:2.2.1

.PHONY: update-conda-env
update-conda-env:
	echo "Exporting $(ENV_NAME)"
	conda env export -n $(ENV_NAME) > $(ENV_NAME)_env.yml

.PHONY: train-autoencoder
train-autoencoder: format
	$(call python_script, scripts/text_autoencoder.py)

.PHONY: partial-train-autoencoder
partial-train-autoencoder:
	DATA_FILE=data/wikipedia_20220220_pt_partial.csv DATASET_SIZE=1000 python scripts/text_autoencoder.py

.PHONY: train-transformer-autoencoder
train-transformer-autoencoder: format
	$(call python_script, scripts/text_autoencoder_transformer.py)

.PHONY: download_wikipedia_dataset
download_wikipedia_dataset:
	python scripts/download_wikipedia_data.py

.PHONY: download_bertimbau_tensorflow_checkpoint
download_bertimbau_tensorflow_checkpoint:
	curl -o $(DATA_DIR)/bertimbau-base-portuguese-cased_tensorflow_checkpoint.zip https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_tensorflow_checkpoint.zip
	curl -o $(DATA_DIR)/bertimbau-base-vocab.txt https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt
	curl -o $(DATA_DIR)/bertimbau-large-portuguese-cased_tensorflow_checkpoint.zip https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/bert-large-portuguese-cased_tensorflow_checkpoint.zip
	curl -o $(DATA_DIR)/bertimbau-large-vocab.txt https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/vocab.txt

.PHONY: show_data_info
show_data_info:
	@echo Data file: $(DATA_FILE)
	@echo Data file size: $(DATASET_SIZE)
	@echo Vocabulary file: $(VOCAB_FILE)
	@echo Vocabulary file size: $(VOCAB_SIZE)
	@echo Wikipedia data dir: $(WIKIPEDIA_DATA_DIR)
	@echo Wikipedia data files: $(WIKIPEDIA_DATA_FILES_COUNT)
	@echo Wikipedia dataset size: $(WIKIPEDIA_DATASET_SIZE)

.PHONY: predict
predict:
	$(call python_script, scripts/predict_text.py)
