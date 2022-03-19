import csv
import os
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed

import transformers
from datasets import load_dataset
import tensorflow as tf


DATA_DIR = os.environ.get("DATA_DIR", "data")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1000))


def has_no_minimum_words(sentence):
    return len(sentence.split(" ")) <= 5


def has_invalid_char(sentence):
    return "|" in sentence


def sentence_segmentation(sample):
    sentences = []
    for text in sample["text"]:
        for sentence in text.split("."):
            sentence = sentence.replace("\n", " ").strip()
            if has_no_minimum_words(sentence) or has_invalid_char(sentence):
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
    datasets = datasets.map(
        sentence_segmentation, batched=True, remove_columns="title", num_proc=6
    )
    return datasets


def dataset_generator():
    dataset_name = "wikipedia"
    dataset_language = "pt"
    dataset_date = "20220220"
    datasets = get_wikipedia_dataset(dataset_name, dataset_language, dataset_date)
    for dataset_name in datasets:
        for batch in datasets[dataset_name].to_dict(batched=True):
            for sample in batch["text"]:
                yield (bytes(sample, "utf8"),)


def write_wikipedia_file():
    print("Writing Wikipedia file.")
    dataset_name = "wikipedia"
    dataset_language = "pt"
    dataset_date = "20220220"
    dataset = get_wikipedia_dataset(dataset_name, dataset_language, dataset_date)
    dataset["train"].to_csv(
        f"{DATA_DIR}/{dataset_name}_{dataset_date}_{dataset_language}.csv",
        num_proc=6,
        quoting=csv.QUOTE_ALL,
        index=False,
    )


def load_sentence_dataset():
    return tf.data.Dataset.from_generator(
        dataset_generator, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),),
    )


def write_tfrecord_file(filepath, batch):
    print(f"Writing {filepath}")
    with tf.io.TFRecordWriter(filepath) as file_writer:
        for sample in batch[0]:
            file_writer.write(serialize_sample(sample.numpy()))
    return f"{filepath} written"


def main():

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(f"{DATA_DIR}/wikipedia"):
        os.makedirs(f"{DATA_DIR}/wikipedia")

    write_wikipedia_file()
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        sentences_dataset = load_sentence_dataset()
        counter = 0
        for batch in sentences_dataset.batch(BATCH_SIZE):
            results.append(
                executor.submit(
                    write_tfrecord_file,
                    filepath=f"{DATA_DIR}/wikipedia/{counter}_wikidata.tfrecords",
                    batch=batch,
                )
            )
            counter += 1

    total_files_written = 0
    for result in results:
        try:
            result_returned = result.result()
            if result_returned:
                total_files_written += 1
                print(result_returned)
        except Exception as e:
            print(e)

    print("Total files written: {total_files_written}")


if __name__ == "__main__":
    main()
