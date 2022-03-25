import csv
import os
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import re

import transformers
from datasets import load_dataset
import tensorflow as tf


DATA_DIR = os.environ.get("DATA_DIR", "data")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1000))
MAX_WORKERS = 10
MINIMUM_SENTENCE_WORD_COUNT = 16

VALID_SENTENCE_REGEX = r"^[\s\w,]+$"


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
    feature = {"text": text_feature}
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def get_wikipedia_dataset(
    dataset_name="wikipedia", dataset_language="pt", dataset_date="20220220"
):
    datasets = load_dataset(
        "wikipedia",
        language=dataset_language,
        date=dataset_date,
        beam_runner="DirectRunner",
    )
    datasets["train"].to_csv(
        f"{DATA_DIR}/wikipedia/all.csv",
        num_proc=MAX_WORKERS,
        quoting=csv.QUOTE_ALL,
        index=False,
    )
    datasets = datasets.map(
        sentence_segmentation, batched=True, remove_columns="title", num_proc=6
    )
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


def load_tf_dataset(dataset):
    def dataset_generator():
        for batch in dataset.to_dict(batched=True):
            for sample in batch["text"]:
                yield (bytes(sample, "utf8"),)

    return tf.data.Dataset.from_generator(
        dataset_generator, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),),
    )


def write_tfrecord_file(filepath, batch):
    print(f"Writing {filepath}")
    with tf.io.TFRecordWriter(filepath) as file_writer:
        for sample in batch[0]:
            file_writer.write(serialize_sample(sample.numpy()))


def write_tfrecord_files(datasets):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for dataset_split_name in datasets:
            sentences_dataset = load_tf_dataset(datasets[dataset_split_name])
            counter = 0
            for batch in sentences_dataset.batch(BATCH_SIZE):
                results.append(
                    executor.submit(
                        write_tfrecord_file,
                        filepath=f"{DATA_DIR}/wikipedia/{dataset_split_name}/{counter}.tfrecords",
                        batch=batch,
                    )
                )
                counter += 1

    total_files_written = 0
    futures_completed = 0
    for result in as_completed(results):
        futures_completed += 1
        try:
            result_returned = result.result()
            total_files_written += 1
        except Exception as e:
            print(e)

    print(f"Futures completed: {futures_completed}")
    print(f"Total files written: {total_files_written}")


def main():

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/train", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/test", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/evaluation", exist_ok=True)

    datasets, metadata = load_datasets()
    write_wikipedia_file(datasets)
    write_tfrecord_files(datasets)

    with open(f"{DATA_DIR}/wikipedia/metadata.json", "w") as metadatafile:
        json.dump(metadata, metadatafile)


if __name__ == "__main__":
    main()
