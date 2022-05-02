import csv
import shutil
import os
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import re

import transformers
from datasets import load_dataset
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors


DATA_DIR = os.environ.get("DATA_DIR", "data")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1000))
MAX_WORKERS = 10
MINIMUM_SENTENCE_WORD_COUNT = 3
save_csv = True
save_raw_dataset = True
max_text_length = 40
embedding_dimensions = 50

PADDING_TOKEN = "<PAD>"
UNK_TOKEN = "<unk>"

VALID_SENTENCE_REGEX = r"^[\s\w,]+$"


tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=UNK_TOKEN)


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


def serialize_tokens(tokens):
    serialized_tokens = tf.io.serialize_tensor(tokens)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_tokens.numpy()])
    )


def serialize_embeddings(embeddings):
    serialized_tokens = tf.io.serialize_tensor(embeddings)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_tokens.numpy()])
    )


def serialize_sample(text, tokens, embeddings):
    text_feature = bytes_feature(text)
    tokens_feature = serialize_tokens(tokens)
    embeddings_feature = serialize_embeddings(embeddings)
    feature = {
        "text": text_feature,
        "tokens": tokens_feature,
        "embeddings": embeddings_feature,
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
    datasets = datasets.map(
        sentence_segmentation, batched=True, remove_columns="title", num_proc=6
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


def get_embedding_sentence(sentence):
    embsentence = []
    for word in sentence:
        if word in embeddingmodel:
            embsentence.append(embeddingmodel.get_vector(word))
        else:
            embsentence.append(embeddingmodel.get_vector(UNK_TOKEN))
    return embsentence


def load_tf_dataset(dataset, tokenize=False):
    def dataset_generator():
        for batch in dataset.to_dict(batched=True):
            for sample in batch["text"]:
                if tokenize:
                    tokens = tf.keras.preprocessing.text.text_to_word_sequence(sample)[
                        :max_text_length
                    ]
                    if len(tokens) < max_text_length:
                        tokens += (max_text_length - len(tokens)) * [PADDING_TOKEN]
                    assert len(tokens) == max_text_length
                    embsentence = get_embedding_sentence(tokens)
                    yield (
                        bytes(sample, "utf8"),
                        [bytes(token, "utf8") for token in tokens],
                        embsentence,
                    )
                else:
                    yield (bytes(sample, "utf8"),)

    signature = (tf.TensorSpec(shape=(), dtype=tf.string),)
    if tokenize:
        signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(max_text_length,), dtype=tf.string),
            tf.TensorSpec(
                shape=(max_text_length, embedding_dimensions), dtype=tf.float32
            ),
        )
    return tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=signature,
    )


def write_tfrecord_file(filepath, batch):
    print(f"Writing {filepath}")
    with tf.io.TFRecordWriter(filepath) as file_writer:
        for sample, tokens, embeddings in zip(batch[0], batch[1], batch[2]):
            file_writer.write(serialize_sample(sample.numpy(), tokens, embeddings))


def write_tfrecord_files(datasets):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for dataset_split_name in datasets:
            sentences_dataset = load_tf_dataset(datasets[dataset_split_name], False)
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


def write_embedding_tfrecord_files(datasets):
    print("Writing datasets tokenized and with embeddings")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for dataset_split_name in datasets:
            print(f"Writing {dataset_split_name} dataset")
            sentences_dataset = load_tf_dataset(datasets[dataset_split_name], True)
            print(sentences_dataset.element_spec)
            print(list(sentences_dataset.take(1)))
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


def decode_fn(encoded_example):
    sample = tf.io.parse_example(
        encoded_example,
        {
            "text": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
            "tokens": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
            "embeddings": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        },
    )
    tokens = tf.io.parse_tensor(sample["tokens"], out_type=tf.string)
    tokens = tf.ensure_shape(tokens, (max_text_length,))
    embeddings = tf.io.parse_tensor(sample["embeddings"], out_type=tf.float32)
    embeddings = tf.ensure_shape(embeddings, (max_text_length, embedding_dimensions))
    return sample["text"], tokens, embeddings


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


def load_embeddings():
    global embeddingmodel
    embeddingmodel = KeyedVectors.load_word2vec_format(
        f"{DATA_DIR}/embeddings/glove_s50.txt"
    )
    embeddingmodel.add_vector(PADDING_TOKEN, np.zeros((50,)))
    embeddingmodel.save_word2vec_format(
        f"{DATA_DIR}/wikipedia/embeddings.txt",
        write_header=True,
    )
    print("Embeddings loaded.")


def main():

    print(f"DATA_DIR: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/train", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/test", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/wikipedia/evaluation", exist_ok=True)

    datasets, metadata = load_datasets()
    fit_tokenizer(datasets)
    if save_csv:
        write_wikipedia_file(datasets)
    if save_raw_dataset:
        load_embeddings()
        write_embedding_tfrecord_files(datasets)
        ds = tf.data.TFRecordDataset(f"{DATA_DIR}/wikipedia/train/0.tfrecords")
        ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        print(ds.element_spec)
        sample = list(ds.take(1))
        print(sample)
        print(
            " ".join(
                [
                    embeddingmodel.similar_by_vector(emb.numpy())[0][0]
                    for emb in sample[0][2]
                ]
            )
        )
    with open(f"{DATA_DIR}/wikipedia/metadata.json", "w") as metadatafile:
        json.dump(metadata, metadatafile)


if __name__ == "__main__":
    main()
