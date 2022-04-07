import os
import logging
import argparse
import pathlib

import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors

from gazettes.data import WikipediaDataset

PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"


def load_model(model_path):
    logging.info(f"Loading model {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
    return model


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=pathlib.Path,
        help="Path to the model to load",
    )
    parser.add_argument(
        "--embeddings-file",
        required=True,
        type=pathlib.Path,
        help="Path to the embeddings file",
    )
    parser.add_argument(
        "--embeddings-dimensions",
        required=True,
        type=int,
        help="Path to the embeddings file",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=pathlib.Path,
        help="Path to dataset directory with the data to predict",
    )
    parser.add_argument(
        "--max-text-length",
        required=False,
        type=int,
        default=40,
        help="Path to dataset directory with the data to predict",
    )
    parser.add_argument(
        "--batch-size", required=False, type=int, default=32, help="",
    )
    parser.add_argument(
        "--parallel-calls", required=False, type=int, default=tf.data.AUTOTUNE, help="",
    )
    args = parser.parse_args()
    args.model = str(args.model)
    args.dataset_dir = str(args.dataset_dir)
    return args


def load_embeddings(embeddings_file, embeddings_dimensions):
    global embeddingmodel
    embeddingmodel = KeyedVectors.load_word2vec_format(embeddings_file)
    embeddingmodel.add_vector(
        PADDING_TOKEN, np.random.uniform(-1, 1, embeddings_dimensions)
    )
    embeddingmodel.add_vector(
        UNKNOWN_TOKEN, np.random.uniform(-1, 1, embeddings_dimensions)
    )


def gen_word_sequence(data_dir: str, batch_size: int):
    train_dataset = WikipediaDataset(
        data_dir.decode("utf-8"),
        parallel_file_read=tf.data.AUTOTUNE,
        batch_size=batch_size,
    )
    for batch in train_dataset.as_numpy_iterator():
        for sentence in batch:
            yield tf.keras.preprocessing.text.text_to_word_sequence(
                sentence.decode("utf-8")
            )


def gen_embedded_dataset(data_dir: str, max_text_length: int, batch_size: int):
    for sentence in gen_word_sequence(data_dir, batch_size):
        embedding_sentence = []
        for word in sentence:
            if word in embeddingmodel:
                embedding_sentence.append(embeddingmodel.get_vector(word))
            else:
                embedding_sentence.append(embeddingmodel.get_vector(UNKNOWN_TOKEN))
        if len(embedding_sentence) > max_text_length:
            embedding_sentence = embedding_sentence[:max_text_length]
        if len(embedding_sentence) < max_text_length:
            for _ in range(max_text_length - len(embedding_sentence)):
                embedding_sentence.append(embeddingmodel.get_vector("<PAD>"))
        assert len(embedding_sentence) == max_text_length
        yield embedding_sentence


def load_embedded_dataset(
    data_dir: str, max_text_length: int, embeddings_dimensions: int, batch_size: int
):
    dataset = tf.data.Dataset.from_generator(
        gen_embedded_dataset,
        output_signature=(
            tf.TensorSpec(
                shape=(max_text_length, embeddings_dimensions), dtype=tf.float32
            )
        ),
        args=(data_dir, max_text_length, batch_size,),
    ).batch(batch_size)
    return dataset


def convert_sentencens_embedding(sentence):
    string = []
    for embedding in sentence:
        string.append(embeddingmodel.similar_by_vector(embedding, 1)[0][0])
    return " ".join(string)


def convert_dataset_to_string(dataset):
    for element in dataset.unbatch().as_numpy_iterator():
        yield convert_sentencens_embedding(element)


def convert_predictions_to_string(predictions):
    for element in predictions:
        yield convert_sentencens_embedding(element)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.DEBUG)
    args = parse_command_line_arguments()
    model = load_model(args.model)
    embeddings = load_embeddings(args.embeddings_file, args.embeddings_dimensions)
    dataset = load_embedded_dataset(
        args.dataset_dir,
        args.max_text_length,
        args.embeddings_dimensions,
        args.batch_size,
    )

    inputt = dataset.take(1)
    predictions = model.predict(inputt)
    for inputt, prediction in zip(
        convert_dataset_to_string(inputt), convert_predictions_to_string(predictions)
    ):
        print(f"INPUT: {inputt}\nPREDICTION: {prediction}\n\n")


if __name__ == "__main__":
    main()
