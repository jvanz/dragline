import os
import logging
import csv

import tensorflow as tf
from tensorflow import keras
import numpy as np

from gazettes.data import WikipediaDataset


WIKIPEDIA_DATA_DIR = str(os.environ.get("WIKIPEDIA_DATA_DIR", "data/wikipedia"))
WIKIPEDIA_DATASET_SIZE = int(os.environ.get("WIKIPEDIA_DATASET_SIZE", 16450980))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 64))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
VOCAB_FILE = str(os.environ["VOCAB_FILE"])
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
if "NUM_PARALLEL_CALLS" in os.environ:
    NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS"))
DIMENSOES_ESPACO_LATENTE = int(os.environ.get("DIMENSOES_ESPACO_LATENTE", 32))
DEFAULT_MODEL_NAME = "text_autoencoder"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)


def get_checkpoint_dir(model):
    checkpoint_dir = f"{os.getcwd()}/checkpoints/{MODEL_NAME}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=16,
                name="embbedding",
                input_length=MAX_TEXT_LENGTH,
            ),
            tf.keras.layers.Dense(
                DIMENSOES_ESPACO_LATENTE, activation="relu", name="encoder1",
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=DIMENSOES_ESPACO_LATENTE, return_sequences=False,
                ),
                name="encoder2",
            ),
            tf.keras.layers.RepeatVector(MAX_TEXT_LENGTH, name="decoder0"),
            tf.keras.layers.Dropout(0.2, name="decoder1"),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=DIMENSOES_ESPACO_LATENTE, return_sequences=True,
                ),
                name="decoder2",
            ),
            tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax", name="decoder3"),
        ],
        name=MODEL_NAME,
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["acc"],
    )
    return model


def create_or_load_model():
    # TODO - load model from checkpoint
    model = create_model()
    model.summary()
    return model


def get_dataset_pertitions(
    dataset, dataset_size, train_split=0.8, evaluation_split=0.1, test_split=0.1,
):
    assert train_split + evaluation_split + test_split == 1

    dataset = dataset.shuffle(dataset_size, seed=7)

    train_size = int(dataset_size * train_split)
    evaluation_size = int(dataset_size * evaluation_split)
    test_size = int(dataset_size * test_split)

    train_dataset = dataset.take(train_size)
    evaluation_dataset = dataset.skip(train_size).take(evaluation_size)
    test_dataset = dataset.skip(train_size + evaluation_size).take(test_size)

    return train_dataset, evaluation_dataset, test_dataset


def train_model(model, train_dataset, validation_dataset, test_dataset):
    logging.info("Training model...")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=get_checkpoint_dir(model),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )
    model.fit(
        train_dataset.batch(
            BATCH_SIZE,
            drop_remainder=True,
            num_parallel_calls=NUM_PARALLEL_CALLS,
            deterministic=False,
        ),
        validation_data=validation_dataset.batch(
            BATCH_SIZE,
            drop_remainder=True,
            num_parallel_calls=NUM_PARALLEL_CALLS,
            deterministic=False,
        ),
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback],
    )
    results = model.evaluate(
        test_dataset.batch(
            BATCH_SIZE,
            drop_remainder=True,
            num_parallel_calls=NUM_PARALLEL_CALLS,
            deterministic=False,
        )
    )
    print()
    print(f"Model evaluation: {results}")
    print()
    model.save("models/text_autoencoder", overwrite=True)


def get_logits(predictions):
    sentences = []
    for sentence in predictions:
        sentence = np.argmax(sentence, axis=1)
        sentences.append(sentence)
    return np.asarray(sentences)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.INFO)

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

    dataset = WikipediaDataset(WIKIPEDIA_DATA_DIR)

    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode="int",
        output_sequence_length=MAX_TEXT_LENGTH,
        name="vectorization_layer",
        vocabulary=VOCAB_FILE,
    )

    def preprocess_text(text):
        vectorized_text = vectorize_layer(text)
        vectorized_target = tf.one_hot(vectorize_layer(text), VOCAB_SIZE)
        return (vectorized_text, vectorized_target)

    dataset = dataset.map(
        preprocess_text, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=False
    )

    def filter_invalid_shapes(text, target):
        return tf.shape(text)[0] == MAX_TEXT_LENGTH

    dataset = dataset.filter(filter_invalid_shapes)
    logging.info(list(dataset.take(1)))

    model = create_or_load_model()
    train_dataset, validation_dataset, test_dataset = get_dataset_pertitions(
        dataset, WIKIPEDIA_DATASET_SIZE
    )
    train_model(model, train_dataset, validation_dataset, test_dataset)
    test_dataset = test_dataset.map(lambda inputt, target: inputt)
    logging.info("Test dataset:")
    logging.info(list(test_dataset.take(1)))

    predictions = model.predict(
        test_dataset.batch(
            BATCH_SIZE,
            drop_remainder=True,
            num_parallel_calls=NUM_PARALLEL_CALLS,
            deterministic=False,
        )
    )
    logging.info(predictions[0])
    logging.info(predictions.shape)
    predictions = get_logits(predictions)
    logging.info(predictions[0])
    string_lookup = tf.keras.layers.StringLookup(vocabulary=VOCAB_FILE, invert=True)
    logging.info(string_lookup(tf.constant(predictions)))



if __name__ == "__main__":
    main()
