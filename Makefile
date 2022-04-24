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

BATCH_SIZE ?= 32
DATA_DIR ?= "data"
EMBEDDING_DIM ?= 50
EMBEDDING_FILE ?= "$(DATA_DIR)/embeddings/glove_s50.txt"
ENV_NAME ?= $(shell conda env export --json | jq ".name")
EPOCHS ?= 1000
MODEL_NAME ?= "text_autoencoder"
MODEL_PATH ?= "$(PWD)/models"
VOCAB_FILE ?= "$(DATA_DIR)/bertimbau_base_vocab.txt"
VOCAB_SIZE ?= $(shell cat $(VOCAB_FILE) | wc -l)
WIKIPEDIA_DATASET_SIZE ?= 1.0
WIKIPEDIA_DATA_DIR ?= "$(DATA_DIR)/wikipedia"
PATIENCE ?= 20
LEARNING_RATE ?= 0.00001


python_script = PYTHONPATH=$(PWD) \
	BATCH_SIZE=$(BATCH_SIZE) \
	EPOCHS=$(EPOCHS) \
	MODEL_NAME=$(MODEL_NAME) \
	MODEL_PATH=$(MODEL_PATH) \
	TF_CPP_MIN_LOG_LEVEL=2 \
	VOCAB_FILE=$(VOCAB_FILE) \
	VOCAB_SIZE=$(VOCAB_SIZE) \
	WIKIPEDIA_DATASET_SIZE=$(WIKIPEDIA_DATASET_SIZE) \
	WIKIPEDIA_DATA_DIR=$(WIKIPEDIA_DATA_DIR) \
	EMBEDDING_FILE=$(EMBEDDING_FILE) \
	EMBEDDING_DIM=$(EMBEDDING_DIM) \
	PATIENCE=$(PATIENCE) \
	LEARNING_RATE=$(LEARNING_RATE) \
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
train-autoencoder: VOCAB_FILE=$(DATA_DIR)/wikipedia_vocab
train-autoencoder:
	rm -rf logs
	PYTHONPATH=$(PWD) python scripts/text_autoencoder.py \
		--rnn-type lstm \
		--hidden-layers-count 1 \
		--embedding-dimensions 50 \
		--embedding-file "$(DATA_DIR)/embeddings/glove_s50.txt" \
		--dataset-dir $(WIKIPEDIA_DATA_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--train


.PHONY: train-transformer-autoencoder
train-transformer-autoencoder: MODEL_NAME="text_transformer_autoencoder"
train-transformer-autoencoder:
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
	@echo Vocabulary file: $(VOCAB_FILE)
	@echo Vocabulary file size: $(VOCAB_SIZE)
	@echo Wikipedia data dir: $(WIKIPEDIA_DATA_DIR)
	@echo Wikipedia dataset size: $(WIKIPEDIA_DATASET_SIZE)

.PHONY: predict-autoencoder
predict-autoencoder: VOCAB_FILE=$(DATA_DIR)/wikipedia_vocab
predict-autoencoder:
	PYTHONPATH=$(PWD) python scripts/predict_text.py \
		   -m models/${MODEL_NAME} \
		   --embeddings-file=$(EMBEDDING_FILE) \
		   --embeddings-dimensions=$(EMBEDDING_DIM) \
		   --dataset-dir=$(WIKIPEDIA_DATA_DIR)/test \
		   --vocab-size=$(VOCAB_SIZE)


.PHONY: build-vocab
build-vocab: VOCAB_FILE=$(DATA_DIR)/wikipedia_vocab
build-vocab:
	$(call python_script, scripts/build_vocabulary.py)


.PHONY: clean-cache
clean-cache:
	rm -rf $(WIKIPEDIA_DATA_DIR)/cache
	rm -rf $(WIKIPEDIA_DATA_DIR)/train/cache
	rm -rf $(WIKIPEDIA_DATA_DIR)/test/cache
	rm -rf $(WIKIPEDIA_DATA_DIR)/evaluation/cache

.PHONY: clean-wikipedia
clean-wikipedia:
	rm -rf $(WIKIPEDIA_DATA_DIR)

.PHONY: download-word-embeddings
download-word-embeddings:
	mkdir -p $(DATA_DIR)/embeddings
	curl -o $(DATA_DIR)/embeddings/glove_s50.zip http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s50.zip
	unzip -d $(DATA_DIR)/embeddings $(DATA_DIR)/embeddings/glove_s50.zip
