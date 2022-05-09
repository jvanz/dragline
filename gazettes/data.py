import json
import re
import os
import csv
from urllib.parse import urlparse
import logging

from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

PRETRAINED_UNK_EMBEDDING_TOKEN = "<unk>"
UNK_TOKEN = "[UNK]"
PADDING_TOKEN = ""


def load_vocabulary(tokenizer_config_file: str, vocabulary_size: int):
    with open(tokenizer_config_file, "r") as config_file:
        config = json.load(config_file)
        vocabulary = list(json.loads(config["config"]["word_counts"]).items())
        vocabulary.sort(key=lambda i: i[1], reverse=True)
        vocabulary = list(map(lambda i: i[0], vocabulary))
        vocabulary = vocabulary[:vocabulary_size]
        # add padding and OOV token
        vocabulary.insert(0, UNK_TOKEN)
        vocabulary.insert(0, PADDING_TOKEN)
        return vocabulary


def load_pretrained_embeddings(embeddings_file: str):
    return KeyedVectors.load_word2vec_format(embeddings_file)


def load_tokenizer(tokenizer_config_file: str):
    with open(tokenizer_config_file, "r") as config_file:
        config = json.load(config_file)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(config))
        return tokenizer


def prepare_embedding_matrix(
    tokenizador_config_file: str, embeddings_file: str, vocabulary_size: int
):
    tokenizer = load_tokenizer(tokenizador_config_file)
    embeddings = load_pretrained_embeddings(embeddings_file)
    # We have the OOV token and padding token. Thus, increase 2 in the vocab size
    embedding_matrix = np.zeros((vocabulary_size + 2, embeddings.vector_size))
    embedding_matrix[1] = embeddings.get_vector(PRETRAINED_UNK_EMBEDDING_TOKEN)

    vocabulary = list(tokenizer.word_counts.items())
    vocabulary.sort(key=lambda i: i[1], reverse=True)
    vocabulary = list(map(lambda i: i[0], vocabulary))
    vocabulary = vocabulary[:vocabulary_size]

    for index, word in enumerate(vocabulary, start=2):
        if embeddings.has_index_for(word):
            embedding = embeddings.get_vector(word)
            embedding_matrix[index] = embedding

    return embedding_matrix


def load_csv_file_column(csvfile_name: str, column: str):
    if type(csvfile_name) is not str:
        csvfile_name = str(csvfile_name, "utf8")
    if type(column) is not str:
        column = str(column, "utf8")
    with open(csvfile_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row[column]


class WikipediaDataset(tf.data.Dataset):
    def __new__(cls, data_dir: str, batch_size: int = 32):
        # datafiles = os.listdir(data_dir)
        # datafiles = list(filter(lambda x: x.endswith("tfrecords"), datafiles))
        # datafiles = [f"{data_dir}/{datafile}" for datafile in datafiles]
        # datafilesdataset = tf.data.Dataset.from_tensor_slices(datafiles)
        # NUM_SHARDS = 6

        # @tf.function
        # def parse_samples(batched_samples):
        #     return tf.io.parse_example(
        #         batched_samples,
        #         {
        #             "text": tf.io.FixedLenFeature(
        #                 [], dtype=tf.string, default_value=""
        #             ),
        #         },
        #     )

        # def make_dataset(shard_index):
        #     filenames = datafilesdataset.shard(NUM_SHARDS, shard_index)
        #     return (
        #         tf.data.TFRecordDataset(filenames)
        #         .batch(batch_size)
        #         .map(
        #             parse_samples,
        #             num_parallel_calls=tf.data.AUTOTUNE,
        #             deterministic=False,
        #         )
        #     )

        # indices = tf.data.Dataset.range(NUM_SHARDS)
        # dataset = indices.interleave(
        #     make_dataset,
        #     num_parallel_calls=tf.data.AUTOTUNE,
        #     deterministic=False,
        # ).prefetch(tf.data.AUTOTUNE)

        return (
            tf.data.Dataset.from_generator(
                load_csv_file_column,
                args=(data_dir, "text"),
                output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


class TextAutoencoderWikipediaDataset(tf.data.Dataset):
    def __new__(
        cls,
        data_dir: str,
        vocabulary,
        batch_size: int = 32,
        max_text_length: int = 40,
    ):
        dataset = WikipediaDataset(data_dir, batch_size=batch_size)
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=len(vocabulary),
            output_mode="int",
            output_sequence_length=max_text_length,
        )
        vectorize_layer.set_vocabulary(vocabulary)

        @tf.function
        def vectorize_text(text):
            target = vectorize_layer(text)
            target = tf.ensure_shape(target, [batch_size, max_text_length])
            target = tf.one_hot(target, len(vocabulary))
            return text, target

        @tf.function
        def ensure_shape(text, target):
            return text, tf.ensure_shape(target, [batch_size, max_text_length])

        @tf.function
        def one_hot(text, target):
            return text, tf.one_hot(target, len(vocabulary))

        dataset = dataset.map(
            vectorize_text, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
        # .map(ensure_shape, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        # .map(one_hot, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
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
