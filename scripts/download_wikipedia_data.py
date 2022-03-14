import csv
import os

import transformers
from datasets import load_dataset
import tensorflow as tf


DATA_DIR = os.environ.get("DATA_DIR", "data/wikipedia")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1000))


def has_no_minimum_words(sentence):
    return len(sentence.split(" ")) <= 3


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


def dataset_generator():
    dataset_name = "wikipedia"
    dataset_language = "pt"
    dataset_date = "20220220"
    datasets = load_dataset(
        "wikipedia",
        language=dataset_language,
        date=dataset_date,
        beam_runner="DirectRunner",
    )
    datasets = datasets.map(
        sentence_segmentation, batched=True, remove_columns="title", num_proc=4
    )
    for dataset_name in datasets:
        for batch in datasets[dataset_name].to_dict(batched=True):
            for sample in batch["text"]:
                yield (bytes(sample, "utf8"),)


def load_sentence_dataset():
    return tf.data.Dataset.from_generator(
        dataset_generator, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),),
    )


def main():
    sentences_dataset = load_sentence_dataset()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    counter = 0
    for batch in sentences_dataset.batch(BATCH_SIZE):
        filepath = f"{DATA_DIR}/{counter}_wikidata.tfrecords"
        with tf.io.TFRecordWriter(filepath) as file_writer:
            for sample in batch[0]:
                file_writer.write(serialize_sample(sample.numpy()))
        counter += 1


if __name__ == "__main__":
    main()
