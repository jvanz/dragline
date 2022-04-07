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
PATIENCE = int(os.environ.get("PATIENCE", 10))
HIDDEN_LAYERS = int(os.environ.get("HIDDEN_LAYERS", 1))
BIDIRECTIONAL = bool(os.environ.get("BIDIRECTIONAL", "1") == "1")
EMBEDDING_DIM = 50

embeddingmodel = KeyedVectors.load_word2vec_format("data/embeddings/glove_s50.txt")
embeddingmodel.add_vector("<PAD>", np.random.uniform(-1, 1, EMBEDDING_DIM))
embeddingmodel.add_vector("<UNK>", np.random.uniform(-1, 1, EMBEDDING_DIM))


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
    embeddind_size = 50

    model = tf.keras.Sequential(name="autoencoder")
    model.add(
        tf.keras.layers.Input(shape=(max_text_length, embeddind_size), name="input")
    )
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=dimensoes_espaco_latente,),
            merge_mode="sum",
            name="encoder",
        )
    )

    model.add(tf.keras.layers.RepeatVector(max_text_length, name="repeater"))
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=embeddind_size, return_sequences=True),
            merge_mode="sum",
            name="decoder",
        )
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanSquaredError()],
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
        # min_delta=1e-2,
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


def gen_word_sequence(data_dir):
    train_dataset = WikipediaDataset(
        data_dir.decode("utf-8"),
        # f"{WIKIPEDIA_DATA_DIR}/train",
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
    partial_load = WIKIPEDIA_DATASET_SIZE
    logging.info("Loading datasets...")
    metadata = load_wikipedia_metadata(WIKIPEDIA_DATA_DIR)
    train_size = int(metadata["train"]["length"] * partial_load)
    logging.info(f"train_size = {train_size}")
    evaluation_size = int(metadata["evaluation"]["length"] * partial_load)
    logging.info(f"evaluation_size = {evaluation_size}")
    test_size = int(metadata["test"]["length"] * partial_load)
    logging.info(f"test_size = {test_size}")

    train_dataset = (
        tf.data.Dataset.from_generator(
            gen_embedded_dataset,
            output_signature=(
                tf.TensorSpec(shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM), dtype=tf.float32)
            ),
            args=(f"{WIKIPEDIA_DATA_DIR}/train",),
        )
        .take(int(train_size / BATCH_SIZE))
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
        .take(int(evaluation_size / BATCH_SIZE))
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
        .take(int(test_size / BATCH_SIZE))
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

    # train_dataset, eval_dataset, test_dataset = load_datasets(WIKIPEDIA_DATASET_SIZE)

    model = create_or_load_model()
    train_model(model, train_dataset, eval_dataset, test_dataset)
    test_predictions(model, test_dataset)


if __name__ == "__main__":
    main()
