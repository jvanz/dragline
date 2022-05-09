import csv
import shutil
import os
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import re

from datasets import load_dataset
import tensorflow as tf
import numpy as np


DATA_DIR = os.environ.get("DATA_DIR", "data2")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5000))
MAX_WORKERS = 10
MINIMUM_SENTENCE_WORD_COUNT = 4
save_csv = True
save_raw_dataset = True
should_fit_tokenizer = True
max_text_length = 40
embedding_dimensions = 50

PADDING_TOKEN = "<PAD>"
UNK_TOKEN = "[UNK]"

VALID_SENTENCE_REGEX = r"^[\s\w,]+$"


tokenizer = tf.keras.preprocessing.text.Tokenizer()


def has_no_minimum_words_count(
    sentence, minimum_word_count=MINIMUM_SENTENCE_WORD_COUNT
):
    return len(sentence.split(" ")) < minimum_word_count


def is_invalid_sentence(sentence):
    return re.match(VALID_SENTENCE_REGEX, sentence) is None


def remove_multiple_whitespaces(sentence):
    return " ".join(sentence.split())


def remove_new_line(sentence):
    return sentence.replace("\n", " ").strip()


def should_reject_sentence(sentence):
    return is_invalid_sentence(sentence) or has_no_minimum_words_count(sentence)


def remove_titles(text):
    sentences = []
    for sentence in text.splitlines():
        sentence = sentence.strip()
        if has_no_minimum_words_count(sentence, 3):
            continue
        sentences.append(sentence)
    return " ".join(sentences)


def sentence_segmentation(sample):
    sentences = []
    for text in sample["text"]:
        text = remove_titles(text)
        text = remove_multiple_whitespaces(text)
        for sentence in text.split(". "):
            if should_reject_sentence(sentence):
                continue
            sentences.append(sentence)
    sample = {"text": sentences}
    return sample


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample(text):
    text_feature = bytes_feature(text)
    feature = {
        "text": text_feature,
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()
    return example


def get_wikipedia_dataset(
    dataset_name="wikipedia", dataset_language="pt", dataset_date="20220220"
):
    datasets = load_dataset(
        "wikipedia",
        language=dataset_language,
        date=dataset_date,
        beam_runner="DirectRunner",
    )
    if save_csv:
        print("Saving all raw sentences")
        datasets["train"].to_csv(
            f"{DATA_DIR}/wikipedia/all.csv",
            num_proc=MAX_WORKERS,
            quoting=csv.QUOTE_ALL,
            index=False,
        )
    print("Sentence segmentation...")
    datasets = datasets.map(
        sentence_segmentation,
        batched=True,
        num_proc=6,
        remove_columns=["title", "url", "id"],
    )
    if save_csv:
        print("Saving all selected sentences.")
        datasets["train"].to_csv(
            f"{DATA_DIR}/wikipedia/all_sentences.csv",
            num_proc=MAX_WORKERS,
            quoting=csv.QUOTE_ALL,
            index=False,
        )
    datasets = datasets["train"].train_test_split(shuffle=True, keep_in_memory=False)
    dataset_eval_test = datasets["test"].train_test_split(
        test_size=0.5, shuffle=False, keep_in_memory=False
    )
    datasets["evaluation"] = dataset_eval_test["train"]
    datasets["test"] = dataset_eval_test["test"]
    return datasets


def load_datasets():
    print("Loading datasets...")
    dataset_name = "wikipedia"
    dataset_language = "pt"
    dataset_date = "20220220"
    datasets = get_wikipedia_dataset(dataset_name, dataset_language, dataset_date)
    metadata = {
        "name": dataset_name,
        "language": dataset_language,
        "date": dataset_date,
    }
    for dataset_split_name in datasets:
        metadata[dataset_split_name] = {"length": len(datasets[dataset_split_name])}
    return datasets, metadata


def write_wikipedia_file(datasets):
    for dataset_split_name in datasets:
        datasets[dataset_split_name].to_csv(
            f"{DATA_DIR}/wikipedia/{dataset_split_name}.csv",
            num_proc=MAX_WORKERS,
            quoting=csv.QUOTE_ALL,
            index=False,
        )


def load_tf_dataset(dataset, tokenize=False):
    def dataset_generator():
        for batch in dataset.to_dict(batched=True):
            for sample in batch["text"]:
                yield bytes(sample, "utf8"),

    signature = (tf.TensorSpec(shape=(None), dtype=tf.string),)
    return tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=signature,
    )


def write_tfrecord_file(filepath, batch):
    print(f"Writing {filepath}")
    with tf.io.TFRecordWriter(filepath) as file_writer:
        for sample in batch[0].numpy():
            file_writer.write(serialize_sample(sample))


def write_tfrecord_files(datasets):
    results = []
    for dataset_split_name in datasets:
        sentences_dataset = load_tf_dataset(datasets[dataset_split_name])
        counter = 0
        for batch in sentences_dataset.batch(BATCH_SIZE):
            write_tfrecord_file(
                filepath=f"{DATA_DIR}/wikipedia/{dataset_split_name}/{counter}.tfrecords",
                batch=batch,
            )
            counter += 1


def fit_tokenizer(datasets):
    print("Fitting tokenizer")

    def text_gen():
        for dataset_split_name in datasets:
            dataset = datasets[dataset_split_name]
            for batch in dataset.to_dict(batched=True):
                for sample in batch["text"]:
                    yield sample

    tokenizer.fit_on_texts(text_gen())
    with open(f"{DATA_DIR}/wikipedia/tokenizer.json", "w") as tokenizerfile:
        tokenizerfile.write(tokenizer.to_json())
    return tokenizer


def main():

    print(f"DATA_DIR: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/train", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/test", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/evaluation", exist_ok=True)

    datasets, metadata = load_datasets()
    tokenizer = fit_tokenizer(datasets)
    if save_csv:
        write_wikipedia_file(datasets)
    if save_raw_dataset:
        write_tfrecord_files(datasets)
    with open(f"{DATA_DIR}/wikipedia/metadata.json", "w") as metadatafile:
        json.dump(metadata, metadatafile)


if __name__ == "__main__":
    main()
