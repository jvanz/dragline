import os
import logging
import argparse
import pathlib

import tensorflow as tf
from gazettes.data import (
    WikipediaDataset,
    TextAutoencoderWikipediaDataset,
    TextBertAutoencoderWikipediaDataset,
)
import numpy as np


def load_model(model_path):
    logging.info(f"Loading model {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
    return model


def load_test_dataset(
    dataset_type: str,
    wikipedia_data_dir: str,
    num_parallel_calls: int,
    batch_size: int,
    max_text_length: int,
    vocab_file: str,
    vocab_size: int,
):
    if dataset_type == "autoencoder":
        return TextAutoencoderWikipediaDataset(
            f"{wikipedia_data_dir}/test",
            parallel_file_read=num_parallel_calls,
            batch_size=batch_size,
            max_text_length=max_text_length,
            vocabulary=vocab_file,
            vocabulary_size=vocab_size,
        )
    else:
        return TextBertAutoencoderWikipediaDataset(
            f"{wikipedia_data_dir}/test",
            parallel_file_read=num_parallel_calls,
            batch_size=batch_size,
            max_text_length=max_text_length,
            vocabulary=vocab_file,
            vocabulary_size=vocab_size,
        )


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
        "-d",
        "--dataset-type",
        required=True,
        help="Dataset type used to test the model. Allowed values: autoencoder, bert",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=pathlib.Path,
        help="Directory where the dataset is stored",
    )
    parser.add_argument(
        "--vocab-file", required=True, type=pathlib.Path, help="Vocabulary file"
    )
    parser.add_argument("--vocab-size", required=True, type=int, help="Vocabulary size")
    parser.add_argument(
        "--num-parallel_calls",
        required=False,
        default=tf.data.AUTOTUNE,
        type=int,
        help="Number of parallel call when processing the dataset",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        default=32,
        type=int,
        help="Batch size used to read the dataset",
    )
    parser.add_argument(
        "--max-text-length",
        required=False,
        type=int,
        default=64,
        help="Maximum sentence length",
    )
    args = parser.parse_args()
    logging.debug(args)
    if args.dataset_type not in ["autoencoder", "bert"]:
        raise Exception(
            f"{args.dataset_type} is not allowed value for the --dataset-type argument!"
        )
    args.dataset_dir = str(args.dataset_dir)
    args.vocab_file = str(args.vocab_file)
    args.model = str(args.model)
    return args


def get_logits(predictions):
    sentences = []
    for sentence in predictions:
        sentence = np.argmax(sentence, axis=1)
        sentences.append(sentence)
    return np.asarray(sentences)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.DEBUG)
    args = parse_command_line_arguments()
    dataset = load_test_dataset(
        args.dataset_type,
        args.dataset_dir,
        args.num_parallel_calls,
        args.batch_size,
        args.max_text_length,
        args.vocab_file,
        args.vocab_size,
    )
    logging.info(dataset.take(1))
    model = load_model(args.model)

    predictions = model.predict(dataset.take(1))
    logging.info(predictions)
    predictions = get_logits(predictions)
    logging.info(predictions[0])
    string_lookup = tf.keras.layers.StringLookup(
        vocabulary=args.vocab_file, invert=True
    )
    predicted_sentences = string_lookup(predictions)
    logging.info(predicted_sentences)


if __name__ == "__main__":
    main()
