import os
import logging
import csv
import argparse
import pathlib

import tensorflow as tf
from tensorflow import keras
import numpy as np
from gensim.models import KeyedVectors

from gazettes.data import (
    TextAutoencoderWikipediaDataset,
    WikipediaDataset,
    load_wikipedia_metadata,
)


def get_checkpoint_dir(model, name):
    checkpoint_dir = f"{os.getcwd()}/checkpoints/{name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def create_model(
    dimensoes_espaco_latent,
    rnn_type,
    hidden_layers_count,
    max_text_length,
    embedding_dimensions,
    dropout,
    bidirectional,
    activation,
):
    logging.info("Creating model...")

    encoder_input = tf.keras.layers.Input(shape=(max_text_length, embedding_dimensions))
    layer = None
    if rnn_type == "lstm":
        layer = tf.keras.layers.LSTM(
            units=dimensoes_espaco_latent, dropout=dropout, activation=activation
        )
    else:
        layer = tf.keras.layers.GRU(
            units=dimensoes_espaco_latent, dropout=dropout, activation=activation
        )
    encoder = None
    if bidirectional:
        encoder = tf.keras.layers.Bidirectional(layer, merge_mode="sum")(encoder_input)
    else:
        encoder = layer(encoder_input)

    decoder = tf.keras.layers.RepeatVector(max_text_length, name="repeater")(encoder)
    if rnn_type == "lstm":
        if bidirectional:
            decoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=embedding_dimensions,
                    return_sequences=True,
                    dropout=dropout,
                    activation=activation,
                ),
                merge_mode="sum",
            )(decoder)
        else:
            decoder = tf.keras.layers.LSTM(
                units=embedding_dimensions,
                return_sequences=True,
                dropout=dropout,
                activation=activation,
            )(decoder)
    else:
        if bidirectional:
            decoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=embedding_dimensions,
                    return_sequences=True,
                    dropout=dropout,
                    activation=activation,
                ),
                merge_mode="sum",
            )(decoder)
        else:
            decoder = tf.keras.layers.GRU(
                units=embedding_dimensions,
                return_sequences=True,
                dropout=dropout,
                activation=activation,
            )(decoder)

    model = tf.keras.Model(encoder_input, decoder)

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    model.compile(
        loss=loss, optimizer=optimizer, metrics=metrics,
    )
    return model


def create_or_load_model(
    dimensoes_espaco_latent,
    rnn_type,
    hidden_layers_count,
    max_text_length,
    embedding_dimensions,
    dropout,
    bidirectional,
    activation,
):
    # TODO - load model from checkpoint
    model = create_model(
        dimensoes_espaco_latent,
        rnn_type,
        hidden_layers_count,
        max_text_length,
        embedding_dimensions,
        dropout,
        bidirectional,
        activation,
    )
    model.summary()
    return model


def save_model(model, model_path: str):
    logging.info(f"Saving model at {model_path}")
    model.save(f"{model_path}", overwrite=True)
    with open(f"{model_path}/model.json", "w") as jsonfile:
        jsonfile.write(model.to_json())
    tf.keras.utils.plot_model(
        model, show_shapes=True, to_file=f"{model_path}/model_plot.png"
    )


def train_model(
    model, train_dataset, validation_dataset, test_dataset, epochs, model_name, patience
):
    logging.info("Training model...")
    checkpoint_dir = get_checkpoint_dir(model, model_name)
    logging.info(f"Checkpoint dir: {checkpoint_dir}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, restore_best_weights=True,
    )
    tb_callback = tf.keras.callbacks.TensorBoard("./logs", update_freq=1)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            model_checkpoint_callback,
            early_stop_callback,
            tb_callback,
            nan_callback,
        ],
    )


def gen_word_sequence(data_dir, batch_size, num_parallel_calls):
    train_dataset = WikipediaDataset(
        data_dir.decode("utf-8"),
        parallel_file_read=num_parallel_calls,
        batch_size=batch_size,
    )
    for batch in train_dataset.as_numpy_iterator():
        for sentence in batch:
            yield tf.keras.preprocessing.text.text_to_word_sequence(
                sentence.decode("utf-8")
            )


def gen_embedded_dataset(data_dir, batch_size, max_text_length, num_parallel_calls):
    for sentence in gen_word_sequence(data_dir, batch_size, num_parallel_calls):
        embedding_sentence = []
        for word in sentence:
            if word in embeddingmodel:
                embedding_sentence.append(embeddingmodel.get_vector(word))
            else:
                embedding_sentence.append(embeddingmodel.get_vector("<UNK>"))
        if len(embedding_sentence) > max_text_length:
            embedding_sentence = embedding_sentence[:max_text_length]
        if len(embedding_sentence) < max_text_length:
            for _ in range(max_text_length - len(embedding_sentence)):
                embedding_sentence.append(embeddingmodel.get_vector("<PAD>"))
        assert len(embedding_sentence) == max_text_length
        yield embedding_sentence


def convert_embedding_sentence_to_string_sentence(sentence):
    strsentence = []
    for emb in sentence:
        word = embeddingmodel.similar_by_vector(emb, 1)[0][0]
        strsentence.append(word)
    return " ".join(strsentence)


def add_target(sentence):
    return (sentence, sentence)


def load_embedded_dataset(
    dataset_dir: str, batch_size, max_text_length, embedding_dim, num_parallel_calls
):
    logging.info("Loading datasets...")
    metadata = load_wikipedia_metadata(dataset_dir)

    train_dataset = (
        tf.data.Dataset.from_generator(
            gen_embedded_dataset,
            output_signature=(
                tf.TensorSpec(shape=(max_text_length, embedding_dim), dtype=tf.float32)
            ),
            args=(
                f"{dataset_dir}/train",
                batch_size,
                max_text_length,
                num_parallel_calls,
            ),
        )
        .batch(batch_size)
        .map(add_target, num_parallel_calls=num_parallel_calls, deterministic=False,)
        .shuffle(batch_size)
    )
    test_dataset = (
        tf.data.Dataset.from_generator(
            gen_embedded_dataset,
            output_signature=(
                tf.TensorSpec(shape=(max_text_length, embedding_dim), dtype=tf.float32)
            ),
            args=(
                f"{dataset_dir}/test",
                batch_size,
                max_text_length,
                num_parallel_calls,
            ),
        )
        .batch(batch_size)
        .map(add_target, num_parallel_calls=num_parallel_calls, deterministic=False,)
        .shuffle(batch_size)
    )
    eval_dataset = (
        tf.data.Dataset.from_generator(
            gen_embedded_dataset,
            output_signature=(
                tf.TensorSpec(shape=(max_text_length, embedding_dim), dtype=tf.float32)
            ),
            args=(
                f"{dataset_dir}/evaluation",
                batch_size,
                max_text_length,
                num_parallel_calls,
            ),
        )
        .batch(batch_size)
        .map(add_target, num_parallel_calls=num_parallel_calls, deterministic=False,)
        .shuffle(batch_size)
    )
    return train_dataset, eval_dataset, test_dataset


def compare_original_and_generated_sentences(inputs, predictions):
    for inputt, prediction in zip(inputs.unbatch().as_numpy_iterator(), predictions):
        inputt = convert_embedding_sentence_to_string_sentence(inputt)
        prediction = convert_embedding_sentence_to_string_sentence(prediction)
        logging.info(f"{inputt} -> {prediction}")


def test_predictions(model, dataset):
    def remove_target_from_dataset(inputt, target):
        return inputt

    dataset = dataset.map(
        remove_target_from_dataset,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False,
    )
    logging.info(dataset.element_spec)

    predictions = model.predict(dataset.take(1))
    compare_original_and_generated_sentences(dataset.take(1), predictions)
    logging.info(dataset.take(1))
    logging.info(predictions[0])
    logging.info(predictions.shape)


def evaluate_model(model, dataset, model_path):
    results = model.evaluate(dataset, return_dict=True,)
    print()
    print(f"Fitted model evaluation: {results}")
    print()
    logging.info(f"Loading model {model_path}")
    model = tf.keras.models.load_model(model_path, compile=True)
    model.summary()
    results = model.evaluate(dataset, return_dict=True,)
    print()
    print(f"Loaded model evaluation: {results}")
    print()


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rnn-type",
        required=True,
        type=str,
        help="RNN type to be used in the hidden layers",
    )
    parser.add_argument(
        "--hidden-layers-count",
        required=True,
        type=int,
        help="The number of hidden layers in the encoder and decoder ",
    )
    parser.add_argument(
        "--train", required=False, action="store_true", help="Train model from scratch",
    )
    parser.add_argument(
        "--save-model-at",
        required=False,
        type=pathlib.Path,
        help="Save model after training in the defined path",
    )
    parser.add_argument(
        "--embedding-file", required=True, type=pathlib.Path, help="",
    )
    parser.add_argument(
        "--embedding-dimensions", required=True, type=int, help="",
    )
    parser.add_argument(
        "--vocab-size", required=False, type=int, default=10000, help="",
    )
    parser.add_argument(
        "--dimensoes-espaco-latent", required=False, type=int, default=256
    )
    parser.add_argument(
        "--bidirectional-hidden-layers",
        required=False,
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument("--max-text-length", required=False, type=int, default=40)
    parser.add_argument("--batch-size", required=False, type=int, default=32)
    parser.add_argument("--epochs", required=False, type=int, default=1000)
    parser.add_argument("--patience", required=False, type=int, default=20)
    parser.add_argument(
        "--num-parallel-calls", required=False, type=int, default=tf.data.AUTOTUNE
    )
    parser.add_argument("--dropout", required=False, type=float, default=0.2)
    parser.add_argument(
        "--model-name", required=False, type=str, default="autoencoder", help="",
    )
    parser.add_argument(
        "--dataset-dir", required=True, type=pathlib.Path, help="",
    )
    parser.add_argument("--activation", required=False, type=str, default="relu")

    args = parser.parse_args()
    args.embedding_file = str(args.embedding_file)
    return args


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.DEBUG)
    args = command_line_args()
    logging.debug("##########################################")
    logging.debug(args)
    logging.debug("##########################################")

    global embeddingmodel
    embeddingmodel = KeyedVectors.load_word2vec_format(
        args.embedding_file, limit=args.vocab_size
    )
    embeddingmodel.add_vector(
        "<PAD>", np.random.uniform(-1, 1, args.embedding_dimensions)
    )
    embeddingmodel.add_vector(
        "<UNK>", np.random.uniform(-1, 1, args.embedding_dimensions)
    )

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Números de GPUs disponíveis: {gpu_count}")

    model = create_or_load_model(
        args.dimensoes_espaco_latent,
        args.rnn_type,
        args.hidden_layers_count,
        args.max_text_length,
        args.embedding_dimensions,
        args.dropout,
        args.bidirectional_hidden_layers,
        args.activation,
    )

    if args.train:
        train_dataset, eval_dataset, test_dataset = load_embedded_dataset(
            args.dataset_dir,
            args.batch_size,
            args.max_text_length,
            args.embedding_dimensions,
            args.num_parallel_calls,
        )
        logging.info(train_dataset.element_spec)
        logging.info(eval_dataset.element_spec)
        logging.info(test_dataset.element_spec)

        train_model(
            model,
            train_dataset,
            eval_dataset,
            test_dataset,
            args.epochs,
            args.model_name,
            args.patience,
        )
        if args.save_model_at:
            save_model(model, args.save_model_at)
        evaluate_model(model, test_dataset, f"{MODEL_PATH}/{model.name}")


if __name__ == "__main__":
    main()
