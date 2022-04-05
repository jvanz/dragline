import os
import logging
import csv

import tensorflow as tf
from tensorflow import keras
import numpy as np

from gazettes.data import (
    TextAutoencoderWikipediaDataset,
    load_wikipedia_metadata,
)


WIKIPEDIA_DATA_DIR = str(os.environ.get("WIKIPEDIA_DATA_DIR", "data/wikipedia"))
WIKIPEDIA_DATASET_SIZE = float(os.environ.get("WIKIPEDIA_DATASET_SIZE", 1.0))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 64))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
VOCAB_FILE = str(os.environ["VOCAB_FILE"])
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS", tf.data.AUTOTUNE))
DIMENSOES_ESPACO_LATENTE = int(os.environ.get("DIMENSOES_ESPACO_LATENTE", 32))
MODEL_NAME = os.environ.get("MODEL_NAME", "text_autoencoder")
MODEL_PATH = os.environ.get("MODEL_PATH", f"models/{MODEL_NAME}")

DROPOUT = float(os.environ.get("DROPOUT", 0.2))
PATIENCE = int(os.environ.get("PATIENCE", 50))
HIDDEN_LAYERS = int(os.environ.get("HIDDEN_LAYERS", 1))
BIDIRECTIONAL = bool(os.environ.get("BIDIRECTIONAL_RNN", 1))
LSTM_ACTIVATION = "relu"


class TextAutoEncoder(tf.keras.Model):
    def __init__(self, dimensoes_espaco_latente, max_text_length, vocab_size, dropout):
        super(TextAutoEncoder, self).__init__()

        self.encoder = tf.keras.Sequential(name="encoder")
        self.encoder.add(tf.keras.layers.Input(shape=(max_text_length,)))
        self.encoder.add(tf.keras.layers.Reshape((max_text_length, 1)))
        # self.encoder.add(
        #     tf.keras.layers.Embedding(
        #         input_dim=vocab_size, output_dim=16, input_length=max_text_length,
        #     )
        # )
        for _ in range(HIDDEN_LAYERS):
            self.encoder.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        units=dimensoes_espaco_latente,
                        return_sequences=True,
                        dropout=dropout,
                        recurrent_dropout=dropout,
                        activation=LSTM_ACTIVATION,
                    ),
                    merge_mode="sum",
                )
            )
        self.encoder.add(
            tf.keras.layers.LSTM(
                units=dimensoes_espaco_latente,
                return_sequences=False,
                name="encoder_output",
            ),
        )

        self.decoder = tf.keras.Sequential(name="decoder")
        self.decoder.add(tf.keras.layers.Input(shape=(dimensoes_espaco_latente,)))
        self.decoder.add(tf.keras.layers.RepeatVector(max_text_length))
        for _ in range(HIDDEN_LAYERS):
            self.decoder.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        units=dimensoes_espaco_latente,
                        return_sequences=True,
                        dropout=dropout,
                        recurrent_dropout=dropout,
                        activation=LSTM_ACTIVATION,
                    ),
                    merge_mode="sum",
                )
            )

        self.decoder.add(tf.keras.layers.Dense(1))
        self.decoder.add(tf.keras.layers.Reshape((max_text_length,)))

    def call(self, inputt):
        outputs = self.decoder(self.encoder(inputt))
        return outputs


def get_checkpoint_dir(model):
    checkpoint_dir = f"{os.getcwd()}/checkpoints/{MODEL_NAME}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def create_model():
    logging.info("Creating model...")

    dimensoes_espaco_latente = DIMENSOES_ESPACO_LATENTE
    max_text_length = MAX_TEXT_LENGTH
    vocab_size = VOCAB_SIZE
    dropout = DROPOUT

    model = tf.keras.Sequential(name="autoencoder")
    model.add(tf.keras.layers.Input(shape=(max_text_length,)))
    model.add(tf.keras.layers.Reshape((max_text_length, 1)))
    for _ in range(HIDDEN_LAYERS):
        if BIDIRECTIONAL:
            layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=dimensoes_espaco_latente,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    activation=LSTM_ACTIVATION,
                ),
                merge_mode="sum",
            )
        else:
            layer = (
                tf.keras.layers.LSTM(
                    units=dimensoes_espaco_latente,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    activation=LSTM_ACTIVATION,
                ),
            )
        model.add(layer)
    model.add(
        tf.keras.layers.LSTM(
            units=dimensoes_espaco_latente,
            return_sequences=False,
            name="encoder_output",
        ),
    )

    model.add(tf.keras.layers.RepeatVector(max_text_length))
    for _ in range(HIDDEN_LAYERS):
        if BIDIRECTIONAL:
            layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=dimensoes_espaco_latente,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    activation=LSTM_ACTIVATION,
                ),
                merge_mode="sum",
            )
        else:
            layer = (
                tf.keras.layers.LSTM(
                    units=dimensoes_espaco_latente,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    activation=LSTM_ACTIVATION,
                ),
            )
        model.add(layer)

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Reshape((max_text_length,)))

    model.compile(
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()],
    )
    return model


def create_or_load_model():
    # TODO - load model from checkpoint
    model = create_model()
    model.summary()
    return model


def save_model(model, model_path: str):
    tf.keras.utils.plot_model(
        model, show_shapes=True, to_file=f"{model_path}/model_plot.png"
    )
    model.save(model_path, overwrite=True)


def train_model(model, train_dataset, validation_dataset, test_dataset):
    logging.info("Training model...")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=get_checkpoint_dir(model),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        mode="min",
        min_delta=1e-2,
        patience=PATIENCE,
        restore_best_weights=True,
    )
    tb_callback = tf.keras.callbacks.TensorBoard("./logs", update_freq=1)

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, early_stop_callback, tb_callback],
    )
    results = model.evaluate(test_dataset, return_dict=True,)
    print()
    print(f"Model evaluation: {results}")
    print()
    save_model(model, MODEL_PATH)


def load_datasets(partial_load: float = 1.0):
    logging.info("Loading datasets...")
    metadata = load_wikipedia_metadata(WIKIPEDIA_DATA_DIR)
    train_size = int(metadata["train"]["length"] * partial_load)
    logging.info(f"train_size = {train_size}")
    evaluation_size = int(metadata["evaluation"]["length"] * partial_load)
    logging.info(f"evaluation_size = {evaluation_size}")
    test_size = int(metadata["test"]["length"] * partial_load)
    logging.info(f"test_size = {test_size}")
    train_dataset = TextAutoencoderWikipediaDataset(
        f"{WIKIPEDIA_DATA_DIR}/train",
        parallel_file_read=NUM_PARALLEL_CALLS,
        batch_size=BATCH_SIZE,
        max_text_length=MAX_TEXT_LENGTH,
        vocabulary=VOCAB_FILE,
        vocabulary_size=VOCAB_SIZE,
    ).take(int(train_size / BATCH_SIZE))
    eval_dataset = TextAutoencoderWikipediaDataset(
        f"{WIKIPEDIA_DATA_DIR}/evaluation",
        parallel_file_read=NUM_PARALLEL_CALLS,
        batch_size=BATCH_SIZE,
        max_text_length=MAX_TEXT_LENGTH,
        vocabulary=VOCAB_FILE,
        vocabulary_size=VOCAB_SIZE,
    ).take(int(evaluation_size / BATCH_SIZE))
    test_dataset = TextAutoencoderWikipediaDataset(
        f"{WIKIPEDIA_DATA_DIR}/test",
        parallel_file_read=NUM_PARALLEL_CALLS,
        batch_size=BATCH_SIZE,
        max_text_length=MAX_TEXT_LENGTH,
        vocabulary=VOCAB_FILE,
        vocabulary_size=VOCAB_SIZE,
    ).take(int(test_size / BATCH_SIZE))
    logging.info("Datasets loaded.")
    return train_dataset, eval_dataset, test_dataset


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.DEBUG)

    logging.info(f"WIKIPEDIA_DATA_DIR = {WIKIPEDIA_DATA_DIR}")
    logging.info(f"WIKIPEDIA_DATASET_SIZE = {WIKIPEDIA_DATASET_SIZE}")
    logging.info(f"MAX_TEXT_LENGTH = {MAX_TEXT_LENGTH}")
    logging.info(f"VOCAB_SIZE = {VOCAB_SIZE}")
    logging.info(f"VOCAB_FILE = {VOCAB_FILE}")
    logging.info(f"BATCH_SIZE = {BATCH_SIZE}")
    logging.info(f"EPOCHS = {EPOCHS}")
    logging.info(f"LEARNING_RATE = {LEARNING_RATE}")
    logging.info(f"NUM_PARALLEL_CALLS = {NUM_PARALLEL_CALLS}")
    logging.info(f"DIMENSOES_ESPACO_LATENTE = {DIMENSOES_ESPACO_LATENTE}")
    logging.info(f"MODEL_NAME = {MODEL_NAME}")

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Números de GPUs disponíveis: {gpu_count}")

    train_dataset, eval_dataset, test_dataset = load_datasets(WIKIPEDIA_DATASET_SIZE)
    logging.debug(list(train_dataset.take(1)))
    logging.info(train_dataset.take(1).take(1).element_spec)

    model = create_or_load_model()
    train_model(model, train_dataset, eval_dataset, test_dataset)
    predictions = model.predict(eval_dataset.take(1))
    logging.info(eval_dataset.take(1))
    logging.info(predictions[0])
    logging.info(predictions.shape)


if __name__ == "__main__":
    main()
