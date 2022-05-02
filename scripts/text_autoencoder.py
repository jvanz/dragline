import os
import logging
import csv
import argparse
import pathlib

import tensorflow as tf
from tensorflow import keras
import numpy as np
from gensim.models import KeyedVectors

from gazettes.data import TextAutoencoderWikipediaDataset

PADDING_TOKEN = "<PAD>"
UNK_TOKEN = "<unk>"


def create_checkpoint_dir(checkpoint_dir, model_name):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(str(checkpoint_dir) + f"/{model_name}", exist_ok=True)


def create_model(
    dimensoes_espaco_latent,
    rnn_type,
    hidden_layers_count,
    max_text_length,
    embedding_dimensions,
    dropout,
    bidirectional,
    activation,
    model_name,
    learning_rate,
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

    model = tf.keras.Model(encoder_input, decoder, name=model_name)

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    metrics = [tf.keras.metrics.MeanSquaredError()]

    model.compile(loss=loss, optimizer=optimizer)
    return model


def get_latest_checkpoint(checkpoint_dir, model_name):
    checkpoint_dir = str(checkpoint_dir) + f"/{model_name}"
    try:
        checkpoints = [
            checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)
        ]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            return latest_checkpoint
    except Exception as err:
        logging.warning(err)
    return None


def create_or_load_model(
    dimensoes_espaco_latent,
    rnn_type,
    hidden_layers_count,
    max_text_length,
    embedding_dimensions,
    dropout,
    bidirectional,
    activation,
    model_name,
    learning_rate,
    from_scratch,
    checkpoint_dir,
):
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir, model_name)
    if from_scratch or latest_checkpoint is None:
        print("Creating a new model")
        model = create_model(
            dimensoes_espaco_latent,
            rnn_type,
            hidden_layers_count,
            max_text_length,
            embedding_dimensions,
            dropout,
            bidirectional,
            activation,
            model_name,
            learning_rate,
        )
    else:
        print("Restoring from", latest_checkpoint)
        model = keras.models.load_model(latest_checkpoint)
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
    model,
    train_dataset,
    validation_dataset,
    epochs,
    model_name,
    patience,
    checkpoint_dir,
):
    logging.info("Training model...")
    create_checkpoint_dir(checkpoint_dir, model_name)
    logging.info(f"Checkpoint dir: {checkpoint_dir}/{model_name}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir)
        + f"/{model_name}/model/{model_name}"
        + "epoch.{epoch:06d}",
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    model_weights_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir)
        + f"/{model_name}/"
        + "weights.{epoch:06d}-{loss:.6f}",
        save_weights_only=False,
        monitor="loss",
        mode="min",
        save_best_only=True,
        save_freq=1000,
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
    )
    tb_callback = tf.keras.callbacks.TensorBoard("./logs", update_freq=1)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            model_checkpoint_callback,
            model_weights_checkpoint_callback,
            early_stop_callback,
            tb_callback,
            nan_callback,
        ],
    )


def load_dataset(
    dataset_dir: str, batch_size, max_text_length, embedding_dim, num_parallel_calls
):
    logging.info("Loading datasets...")

    train_dataset = TextAutoencoderWikipediaDataset(
        f"{dataset_dir}/train", batch_size=batch_size
    ).prefetch(tf.data.AUTOTUNE)
    eval_dataset = TextAutoencoderWikipediaDataset(
        f"{dataset_dir}/evaluation", batch_size=batch_size
    ).prefetch(tf.data.AUTOTUNE)
    test_dataset = TextAutoencoderWikipediaDataset(
        f"{dataset_dir}/test", batch_size=batch_size
    ).prefetch(tf.data.AUTOTUNE)
    return train_dataset, eval_dataset, test_dataset


def evaluate_model(dataset, model_path):
    logging.info(f"Loading model {model_path}")
    model = tf.keras.models.load_model(model_path, compile=True)
    model.summary()
    results = model.evaluate(
        dataset,
        return_dict=True,
    )
    print()
    print(f"Loaded model evaluation: {results}")
    print()


def predict_text(dataset, model_path):
    logging.info(f"Loading model {model_path}")
    model = tf.keras.models.load_model(model_path, compile=True)
    model.summary()
    dataset = dataset.map(
        lambda inputt, output: inputt,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    inputt = list(dataset.unbatch().take(1))[0]
    print(inputt)
    output = model.predict_on_batch(inputt)
    print(output)
    emnsentence = []
    for i, emb in enumerate(output[0]):
        matches = embeddingmodel.similar_by_vector(emb)
        emnsentence.append(matches[0][0])
    print(" ".join(emnsentence))


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
        "--train",
        required=False,
        action="store_true",
        help="Train model from scratch",
    )
    parser.add_argument(
        "--evaluate",
        required=False,
        action="store_true",
        help="Evaluate model saved at --save-model-at",
    )
    parser.add_argument(
        "--predict",
        required=False,
        action="store_true",
        help="Predict some text with the model saved at --save-model-at",
    )
    parser.add_argument(
        "--from-scratch",
        required=False,
        action="store_true",
        help="Start a training from scratch",
    )
    parser.add_argument(
        "--save-model-at",
        required=False,
        type=pathlib.Path,
        help="Save model after training in the defined path",
    )
    parser.add_argument(
        "--embedding-file",
        required=True,
        type=pathlib.Path,
        help="",
    )
    parser.add_argument(
        "--embedding-dimensions",
        required=True,
        type=int,
        help="",
    )
    parser.add_argument(
        "--vocab-size",
        required=False,
        type=int,
        default=10000,
        help="",
    )
    parser.add_argument(
        "--dimensoes-espaco-latent", required=False, type=int, default=256
    )
    parser.add_argument(
        "--bidirectional-hidden-layers",
        required=False,
        action="store_true",
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
        "--model-name",
        required=False,
        type=str,
        default="autoencoder",
        help="",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=pathlib.Path,
        help="",
    )
    parser.add_argument("--activation", required=False, type=str, default="relu")
    parser.add_argument("--learning-rate", required=False, type=float, default=0.001)
    parser.add_argument(
        "--checkpoint-dir",
        required=False,
        type=pathlib.Path,
        default="checkpoints",
        help="",
    )

    args = parser.parse_args()
    args.embedding_file = str(args.embedding_file)
    return args


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.DEBUG)
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Números de GPUs disponíveis: {gpu_count}")
    args = command_line_args()
    logging.debug("##########################################")
    logging.debug(args)
    logging.debug("##########################################")

    train_dataset, eval_dataset, test_dataset = load_dataset(
        args.dataset_dir,
        args.batch_size,
        args.max_text_length,
        args.embedding_dimensions,
        args.num_parallel_calls,
    )
    logging.info(train_dataset.element_spec)
    logging.info(eval_dataset.element_spec)
    logging.info(test_dataset.element_spec)

    if args.train:
        model = create_or_load_model(
            args.dimensoes_espaco_latent,
            args.rnn_type,
            args.hidden_layers_count,
            args.max_text_length,
            args.embedding_dimensions,
            args.dropout,
            args.bidirectional_hidden_layers,
            args.activation,
            args.model_name,
            args.learning_rate,
            args.from_scratch,
            args.checkpoint_dir,
        )
        train_model(
            model,
            train_dataset,
            eval_dataset,
            args.epochs,
            args.model_name,
            args.patience,
            args.checkpoint_dir,
        )
        if args.save_model_at:
            save_model(model, args.save_model_at)

    if args.evaluate:
        evaluate_model(test_dataset, args.save_model_at)

    if args.predict:
        global embeddingmodel
        embeddingmodel = KeyedVectors.load_word2vec_format(args.embedding_file)
        predict_text(test_dataset, args.save_model_at)


if __name__ == "__main__":
    main()
