import json
import re
import os
import csv
from urllib.parse import urlparse
import logging

from bs4 import BeautifulSoup
from transformers import AutoTokenizer

import tensorflow as tf


def decode_fn(encoded_example):
    return tf.io.parse_example(
        encoded_example,
        {"text": tf.io.FixedLenFeature([], dtype=tf.string, default_value="")},
    )["text"]


def get_cache_dir(fallback_dir: str = "", cache_subdirectory: str = ""):
    if not cache_subdirectory:
        raise "Missing cache subdirectory"
    cache_dir = None
    if fallback_dir:
        cache_dir = os.environ.get(
            "CACHE_DIR", f"{fallback_dir}/cache/{cache_subdirectory}"
        )
    else:
        cache_dir = os.environ["CACHE_DIR"]
        cache_dir = f"{cache_dir}/{cache_subdirectory}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class WikipediaDataset(tf.data.Dataset):
    def __new__(cls, data_dir: str, parallel_file_read=4, batch_size=32):
        datafiles = os.listdir(data_dir)
        datafiles = list(filter(lambda x: x.endswith("tfrecords"), datafiles))
        datafiles = [f"{data_dir}/{datafile}" for datafile in datafiles]
        dataset = tf.data.Dataset.from_tensor_slices(datafiles)
        dataset = dataset.interleave(
            lambda datafile: tf.data.TFRecordDataset(datafile),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).map(decode_fn)
        if has_cache_enable():
            dataset.cache(get_cache_dir(data_dir, f"cache_load_data"))
        return dataset


class TextAutoencoderWikipediaDataset(tf.data.Dataset):
    def __new__(
        cls,
        data_dir: str,
        parallel_file_read: int = 4,
        batch_size: int = 32,
        max_text_length: int = 64,
        vocabulary: str = None,
        vocabulary_size: int = 0,
        num_parallel_calls: int = tf.data.AUTOTUNE,
    ):
        dataset = WikipediaDataset(data_dir, batch_size=batch_size)

        vectorize_layer = tf.keras.layers.TextVectorization(
            output_mode="int",
            output_sequence_length=max_text_length,
            name="vectorization_layer",
            vocabulary=vocabulary,
            max_tokens=vocabulary_size,
        )

        def preprocess_text(text):
            vectorized_text = vectorize_layer(text)
            vectorized_target = tf.one_hot(vectorize_layer(text), vocabulary_size)
            return (vectorized_text, vectorized_target)

        dataset = dataset.map(
            preprocess_text, num_parallel_calls=num_parallel_calls, deterministic=False,
        )
        if has_cache_enable():
            dataset.cache(
                get_cache_dir(data_dir, f"cache_text_autoencoder_preprocessing"),
            )

        dataset.vectorize_layer = vectorize_layer
        return dataset


def load_bert_tokenizer(model_checkpoint: str):
    return AutoTokenizer.from_pretrained(
        model_checkpoint, do_lower_case=False, use_fast=False
    )


class TextBertAutoencoderWikipediaDataset(tf.data.Dataset):
    def __new__(
        cls,
        data_dir: str,
        parallel_file_read: int = 4,
        batch_size: int = 1000,
        max_text_length: int = 64,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        model_checkpoint: str = "neuralmind/bert-base-portuguese-cased",
    ):
        dataset = WikipediaDataset(data_dir)
        tokenizer = load_bert_tokenizer(model_checkpoint)

        def preprocess_text(text):
            tokenizer_output = tokenizer(
                text.numpy().decode("utf8"),
                padding="max_length",
                truncation=True,
                max_length=max_text_length,
            )
            return (
                tokenizer_output["input_ids"],
                tokenizer_output["token_type_ids"],
                tokenizer_output["attention_mask"],
            )

        def tf_python_preprocess_text(text):
            preprocessed_text = tf.py_function(
                preprocess_text,
                [text],
                [
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="input_ids"
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="token_type_ids",
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="attention_mask",
                    ),
                ],
            )
            return [
                tf.reshape(tensor, [max_text_length,]) for tensor in preprocessed_text
            ]

        def tf_preprocess_text(batch):
            return tf.map_fn(
                fn=tf_python_preprocess_text,
                elems=batch,
                fn_output_signature=[
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="input_ids"
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="token_type_ids",
                    ),
                    tf.TensorSpec(
                        shape=(max_text_length,), dtype=tf.int32, name="attention_mask",
                    ),
                ],
            )

        dataset = dataset.map(
            tf_preprocess_text,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
        if has_cache_enable():
            dataset.cache(get_cache_dir(data_dir, "transformer_preprocessing"),)

        def organize_targets(input_ids, token_type_ids, attention_mask):
            return (
                (input_ids, token_type_ids, attention_mask),
                target,
                # tf.one_hot(target, vocabulary_size),
            )

        def onehot_target(inputs, target):
            return (
                inputs,
                tf.one_hot(input_ids, tokenizer.vocab_size),
            )

        dataset = dataset.map(organize_targets)
        logging.info(dataset.element_spec)
        dataset = dataset.map(onehot_target)
        if has_cache_enable():
            dataset.cache(get_cache_dir(data_dir, "transformer_one_hot_target"))
        return dataset


def load_wikipedia_metadata(data_dir: str):
    with open(f"{data_dir}/metadata.json", "r") as metadatafile:
        return json.load(metadatafile)


def load_gazettes_csv():
    with open("data/gazettes_unix.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, dialect="unix")
        for row in reader:
            yield row


def load_gazettes_sample():
    with open("data/gazettes_sample.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, dialect="unix")
        for row in reader:
            file_name_from_url = urlparse(row["file_link"]).path.rsplit("/", 1)[-1]
            row["file_path"] = f"data/files/{file_name_from_url}"
            yield row


def get_file_to_store_extracted_text(file_path: str):
    """Generates the file name used to store the file's content

    :file_path: original file path
    :returns: file path used to store the file's content
    """
    file_name = os.path.basename(os.path.splitext(file_path)[0])
    return f"data/files/{file_name}.xml"


def get_file_to_store_clean_text(file_path: str):
    """Generates the file name used to store the file's content

    :file_path: original file path
    :returns: file path used to store the file's content
    """
    file_name = os.path.basename(os.path.splitext(file_path)[0])
    return f"data/files/{file_name}_text.json"


def load_gazette_text(gazette):
    """Load file with the clean text of the given gazette.

    :gazette: gazette which the user wants to get the text
    :returns: list of the sentences in the text
    """
    with open(get_file_to_store_clean_text(gazette["file_path"]), "r") as text_file:
        return json.load(text_file)
    return []


def is_there_clean_content_file(gazette):
    """Check if there is a file with the clean text from the given gazette.

    :returns: true if the file with the clan text. Otherwise, returns false.
    """
    clean_text_file = get_file_to_store_clean_text(gazette["file_path"])
    return os.path.exists(clean_text_file)


def remove_repeating_whitespaces_and_new_lines(text: str) -> str:
    """Removes repeating new lines and tabular char."""
    return re.sub(r"(\n|\r|\t| ){1,}", " ", text)


def remove_repeating_dashes(text: str) -> str:
    """Removes repeating dashes."""
    return re.sub(r"(-|_|\.){1,}", "", text)


def clean_sentence(sentence: str) -> str:
    """Clean sentence removing unnecessary chars."""
    sentence = remove_repeating_whitespaces_and_new_lines(sentence)
    sentence = remove_repeating_dashes(sentence)
    return sentence.lower()


def clean_gazette_text(content_file: str, clean_gazette_text: str):
    """Clean text from the given gazette's content_file and store the clean text
    in the clean_gazette_text

    :content_file: file with the original gazette's text
    :clean_gazette_text: file where the clean text will be stored

    """
    text = []
    with open(content_file, "r") as file:
        soup = BeautifulSoup(file, features="lxml")
        for sentence in soup.stripped_strings:
            if len(sentence) < 3:
                continue
            text.append(clean_sentence(sentence))
    if len(text) == 0:
        raise Exception(f"Could not get text from {content_file}")
    with open(clean_gazette_text, "w") as clean_text_file:
        json.dump(text, clean_text_file)


def clean_gazette_text_if_necessary(gazette, force_clean=False):
    """Clean gazette's text if necessary or if force_clean is true

    The clean text is stored in a file. If the file exists and the force_clean
    is false, this function does nothing.

    :gazette: info  of the gazette which may be clean
    :force_clean: force reclean the text
    """
    if not is_there_clean_content_file(gazette) or force_clean:
        content_file = get_file_to_store_extracted_text(gazette["file_path"])
        clean_text_file = get_file_to_store_clean_text(gazette["file_path"])
        clean_gazette_text(content_file, clean_text_file)


def sample_gazettes_texts(force_clean=False):
    """
    Walk through the gazettes files and return the preprocessed string.
    """
    for gazette in load_gazettes_sample():
        clean_gazette_text_if_necessary(gazette, force_clean=force_clean)
        text = load_gazette_text(gazette)
        yield gazette, text


def has_cache_enable():
    return False
