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

.PHONY: download-models
download-models:
	python -m spacy download pt_core_news_lg
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
