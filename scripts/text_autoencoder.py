import os
import logging
import csv

import tensorflow as tf
from tensorflow import keras


DATA_FILE = str(os.environ.get("DATA_FILE", "data/wikipedia_20220220_pt.csv"))
DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 16450980))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 64))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS", 4))
DIMENSOES_ESPACO_LATENTE = int(os.environ.get("DIMENSOES_ESPACO_LATENTE", 32))
DEFAULT_MODEL_NAME = "text_autoencoder"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)


def load_data_from_file():
    with open(DATA_FILE, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row["text"], row["text"]


def get_checkpoint_dir(model):
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", model.name)
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
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=DIMENSOES_ESPACO_LATENTE, return_sequences=True,
                ),
                name="decoder1",
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(5 * VOCAB_SIZE, activation="relu"),
                name="decoder2",
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax"),
                name="decoder3",
            ),
        ],
        name=MODEL_NAME,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
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
            BATCH_SIZE, drop_remainder=True, num_parallel_calls=6, deterministic=False
        ),
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback],
    )
    results = model.evaluate(test_dataset)
    print()
    print(f"Model evaluation: {results}")
    print()
    model.save("models/text_autoencoder", overwrite=True)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.INFO)

    logging.info(f"DATA_FILE = {DATA_FILE}")
    logging.info(f"DATASET_SIZE = {DATASET_SIZE}")
    logging.info(f"MAX_TEXT_LENGTH = {MAX_TEXT_LENGTH}")
    logging.info(f"VOCAB_SIZE = {VOCAB_SIZE}")
    logging.info(f"BATCH_SIZE = {BATCH_SIZE}")
    logging.info(f"EPOCHS = {EPOCHS}")
    logging.info(f"LEARNING_RATE = {LEARNING_RATE}")
    logging.info(f"NUM_PARALLEL_CALLS = {NUM_PARALLEL_CALLS}")
    logging.info(f"DIMENSOES_ESPACO_LATENTE = {DIMENSOES_ESPACO_LATENTE}")
    logging.info(f"MODEL_NAME = {MODEL_NAME}")

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Números de GPUs disponíveis: {gpu_count}")

    dataset = tf.data.Dataset.from_generator(
        load_data_from_file,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_TEXT_LENGTH,
        name="vectorization_layer",
    )

    logging.info(f"Adapting TextVectorization layers")
    vectorize_layer.adapt(dataset.map(lambda text, target: text))

    def preprocess_text(text, target):
        vectorized_text = vectorize_layer(text)
        vectorized_target = vectorize_layer(target)
        return (vectorized_text, vectorized_target)

    logging.info(f"Preprocessing dataset")
    dataset = dataset.map(
        preprocess_text, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=True
    )
    logging.info(list(dataset.take(1)))

    model = create_or_load_model()
    train_dataset, validation_dataset, test_dataset = get_dataset_pertitions(
        dataset, DATASET_SIZE
    )
    train_model(model, train_dataset, validation_dataset, test_dataset)


if __name__ == "__main__":
    main()
