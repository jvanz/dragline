import os
import logging
import csv

import tensorflow as tf
from tensorflow import keras
import numpy as np
from gensim.models import KeyedVectors

from gazettes.data import (
    TextAutoencoderWikipediaDataset,
    WikipediaDataset,
    load_wikipedia_metadata,
)


BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
BIDIRECTIONAL = bool(os.environ.get("BIDIRECTIONAL", "1") == "1")
DIMENSOES_ESPACO_LATENTE = int(os.environ.get("DIMENSOES_ESPACO_LATENTE", 32))
DROPOUT = float(os.environ.get("DROPOUT", 0.2))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 50))
EMBEDDING_FILE = os.environ["EMBEDDING_FILE"]
EPOCHS = int(os.environ.get("EPOCHS", 10))
HIDDEN_LAYERS = int(os.environ.get("HIDDEN_LAYERS", 1))
HIDDEN_LAYERS_TYPE = str(os.environ.get("HIDDEN_LAYERS_TYPE", "lstm")).lower()
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 64))
NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS", tf.data.AUTOTUNE))
PATIENCE = int(os.environ.get("PATIENCE", 10))
VOCAB_FILE = str(os.environ["VOCAB_FILE"])
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
WIKIPEDIA_DATASET_SIZE = float(os.environ.get("WIKIPEDIA_DATASET_SIZE", 1.0))
WIKIPEDIA_DATA_DIR = str(os.environ.get("WIKIPEDIA_DATA_DIR", "data/wikipedia"))
OPTIMIZER = str(os.environ.get("OPTIMIZER", "adam"))
LOSS = str(os.environ.get("LOSS", "mse"))
METRICS = str(os.environ.get("METRICS", "mse,"))

MODEL_NAME = os.environ.get("MODEL_NAME", "text_autoencoder")
MODEL_NAME = f"{MODEL_NAME}_{HIDDEN_LAYERS}_{'bidirectional_' if BIDIRECTIONAL else ''}{HIDDEN_LAYERS_TYPE}_{VOCAB_SIZE}"
MODEL_PATH = os.environ.get("MODEL_PATH", f"models")

embeddingmodel = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, limit=VOCAB_SIZE)
embeddingmodel.add_vector("<PAD>", np.random.uniform(-1, 1, EMBEDDING_DIM))
embeddingmodel.add_vector("<UNK>", np.random.uniform(-1, 1, EMBEDDING_DIM))


def get_checkpoint_dir(model):
    checkpoint_dir = f"{os.getcwd()}/checkpoints/{build_model_name()}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def build_model_name():
    return f"{MODEL_NAME}_{LOSS}_{OPTIMIZER}_{METRICS.replace(',','-')}"


def create_model():
    logging.info("Creating model...")

    model = tf.keras.Sequential(name=build_model_name())
    model.add(
        tf.keras.layers.Input(shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM), name="input")
    )
    for li in range(HIDDEN_LAYERS):
        layer = None
        layer_name = f"encoder{li}-{HIDDEN_LAYERS_TYPE}"
        if HIDDEN_LAYERS_TYPE == "lstm":
            layer = tf.keras.layers.LSTM(
                units=DIMENSOES_ESPACO_LATENTE,
                dropout=DROPOUT,
                return_sequences=False if li == HIDDEN_LAYERS - 1 else True,
                name=layer_name,
            )
        else:
            layer = tf.keras.layers.GRU(
                units=DIMENSOES_ESPACO_LATENTE,
                dropout=DROPOUT,
                return_sequences=False if li == HIDDEN_LAYERS - 1 else True,
                name=layer_name,
            )
        if BIDIRECTIONAL:
            model.add(
                tf.keras.layers.Bidirectional(layer, name=layer_name, merge_mode="sum")
            )
        else:
            model.add(layer)

    model.add(tf.keras.layers.RepeatVector(MAX_TEXT_LENGTH, name="repeater"))
    for li in range(HIDDEN_LAYERS):
        layer_name = f"decoder{li}-{HIDDEN_LAYERS_TYPE}"
        if HIDDEN_LAYERS_TYPE == "lstm":
            if BIDIRECTIONAL:
                model.add(
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(
                            units=EMBEDDING_DIM, return_sequences=True, dropout=DROPOUT,
                        ),
                        name=layer_name,
                        merge_mode="sum",
                    )
                )
            else:
                model.add(
                    tf.keras.layers.LSTM(
                        units=EMBEDDING_DIM,
                        return_sequences=True,
                        dropout=DROPOUT,
                        name=layer_name,
                    ),
                )
        else:
            if BIDIRECTIONAL:
                model.add(
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.GRU(
                            units=EMBEDDING_DIM, return_sequences=True, dropout=DROPOUT
                        ),
                        name=layer_name,
                        merge_mode="sum",
                    )
                )
            else:
                model.add(
                    tf.keras.layers.GRU(
                        units=EMBEDDING_DIM,
                        return_sequences=True,
                        dropout=DROPOUT,
                        name=layer_name,
                    ),
                )

    loss = None
    if LOSS == "cosine":
        loss = tf.keras.losses.CosineSimilarity()
    else:
        loss = tf.keras.losses.MeanSquaredError()
    logging.info(f"loss: {loss}")

    optimizer = None
    if OPTIMIZER == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    logging.info(f"optimizer: {optimizer}")

    metrics = []
    for metric in METRICS.split(","):
        if metric == "mse":
            metrics.append(tf.keras.metrics.MeanSquaredError())
        if metric == "cosine":
            metrics.append(tf.keras.metrics.CosineSimilarity())
    logging.info(f"metrics: {metrics}")

    model.compile(
        loss=loss, optimizer=optimizer, metrics=metrics,
    )
    return model


def create_or_load_model():
    # TODO - load model from checkpoint
    model = create_model()
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


def train_model(model, train_dataset, validation_dataset, test_dataset):
    logging.info("Training model...")
    checkpoint_dir = get_checkpoint_dir(model)
    logging.info(f"Checkpoint dir: {checkpoint_dir}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=PATIENCE, restore_best_weights=True,
    )
    tb_callback = tf.keras.callbacks.TensorBoard("./logs", update_freq=1)
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            model_checkpoint_callback,
            early_stop_callback,
            tb_callback,
            nan_callback,
        ],
    )


def gen_word_sequence(data_dir):
    train_dataset = WikipediaDataset(
        data_dir.decode("utf-8"),
        parallel_file_read=NUM_PARALLEL_CALLS,
        batch_size=BATCH_SIZE,
    )
    for batch in train_dataset.as_numpy_iterator():
        for sentence in batch:
            yield tf.keras.preprocessing.text.text_to_word_sequence(
                sentence.decode("utf-8")
            )


def gen_embedded_dataset(data_dir):
    for sentence in gen_word_sequence(data_dir):
        embedding_sentence = []
        for word in sentence:
            if word in embeddingmodel:
                embedding_sentence.append(embeddingmodel.get_vector(word))
            else:
                embedding_sentence.append(embeddingmodel.get_vector("<UNK>"))
        if len(embedding_sentence) > MAX_TEXT_LENGTH:
            embedding_sentence = embedding_sentence[:MAX_TEXT_LENGTH]
        if len(embedding_sentence) < MAX_TEXT_LENGTH:
            for _ in range(MAX_TEXT_LENGTH - len(embedding_sentence)):
                embedding_sentence.append(embeddingmodel.get_vector("<PAD>"))
        assert len(embedding_sentence) == MAX_TEXT_LENGTH
        yield embedding_sentence


def convert_embedding_sentence_to_string_sentence(sentence):
    strsentence = []
    for emb in sentence:
        word = embeddingmodel.similar_by_vector(emb, 1)[0][0]
        strsentence.append(word)
    return " ".join(strsentence)


def add_target(sentence):
    return (sentence, sentence)


def load_embedded_dataset():
    logging.info("Loading datasets...")
    metadata = load_wikipedia_metadata(WIKIPEDIA_DATA_DIR)

    train_dataset = (
        tf.data.Dataset.from_generator(
            gen_embedded_dataset,
            output_signature=(
                tf.TensorSpec(shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM), dtype=tf.float32)
            ),
            args=(f"{WIKIPEDIA_DATA_DIR}/train",),
        )
        .batch(BATCH_SIZE)
        .map(add_target, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=False,)
    )
    test_dataset = (
        tf.data.Dataset.from_generator(
            gen_embedded_dataset,
            output_signature=(
                tf.TensorSpec(shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM), dtype=tf.float32)
            ),
            args=(f"{WIKIPEDIA_DATA_DIR}/test",),
        )
        .batch(BATCH_SIZE)
        .map(add_target, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=False,)
    )
    eval_dataset = (
        tf.data.Dataset.from_generator(
            gen_embedded_dataset,
            output_signature=(
                tf.TensorSpec(shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM), dtype=tf.float32)
            ),
            args=(f"{WIKIPEDIA_DATA_DIR}/evaluation",),
        )
        .batch(BATCH_SIZE)
        .map(add_target, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=False,)
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
    logging.info(f"MODEL_PATH = {MODEL_PATH}")
    logging.info(f"DROPOUT = {DROPOUT}")
    logging.info(f"PATIENCE = {PATIENCE}")
    logging.info(f"HIDDEN_LAYERS = {HIDDEN_LAYERS}")
    logging.info(f"BIDIRECTIONAL = {BIDIRECTIONAL}")

    train_dataset, eval_dataset, test_dataset = load_embedded_dataset()
    logging.info(train_dataset.element_spec)
    logging.info(eval_dataset.element_spec)
    logging.info(test_dataset.element_spec)

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Números de GPUs disponíveis: {gpu_count}")

    model = create_or_load_model()
    train_model(model, train_dataset, eval_dataset, test_dataset)
    save_model(model, f"{MODEL_PATH}/{model.name}")
    evaluate_model(model, test_dataset, f"{MODEL_PATH}/{model.name}")
    # test_predictions(model, test_dataset)


if __name__ == "__main__":
    main()
