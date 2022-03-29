import os
import logging

from transformers import (
    TFBertModel,
    AutoTokenizer,
)
import tensorflow as tf
import numpy as np

from gazettes.data import TextBertAutoencoderWikipediaDataset, load_wikipedia_metadata

WIKIPEDIA_DATA_DIR = str(os.environ.get("WIKIPEDIA_DATA_DIR", "data/wikipedia"))
WIKIPEDIA_DATASET_SIZE = float(os.environ.get("WIKIPEDIA_DATASET_SIZE", 1.0))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 64))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS", tf.data.AUTOTUNE))
DIMENSOES_ESPACO_LATENTE = int(os.environ.get("DIMENSOES_ESPACO_LATENTE", 32))
DEFAULT_MODEL_NAME = "text_transformer_autoencoder"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT", "neuralmind/bert-base-portuguese-cased"
)
VOCAB_FILE = os.environ.get("VOCAB_FILE", "data/bertimbau_base_vocab.txt")


def load_bertimbau_model():
    return TFBertModel.from_pretrained(MODEL_CHECKPOINT, from_pt=True)


def create_model():
    # encoder
    bertimbau = load_bertimbau_model()
    bertimbau.trainable = False

    input_ids = tf.keras.layers.Input(
        shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name="input_ids"
    )
    token_types_ids = tf.keras.layers.Input(
        shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name="token_types_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name="attention_mask"
    )

    encoded = bertimbau(
        {
            "input_ids": input_ids,
            "token_types_ids": token_types_ids,
            "attention_mask": attention_mask,
        }
    )["pooler_output"]
    encoder_output = tf.keras.layers.Dense(
        DIMENSOES_ESPACO_LATENTE, name="encoder_output"
    )(encoded)

    # decoder
    decoder = tf.keras.layers.RepeatVector(MAX_TEXT_LENGTH, name="decoder0")(
        encoder_output
    )
    decoder = tf.keras.layers.Dropout(0.2, name="decoder1")(decoder)
    decoder = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=DIMENSOES_ESPACO_LATENTE, return_sequences=True,),
        name="decoder2",
    )(decoder)
    decoder = tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax", name="decoder4")(
        decoder
    )

    model = tf.keras.models.Model(
        inputs=[input_ids, token_types_ids, attention_mask],
        outputs=decoder,
        name=MODEL_NAME,
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["acc"],
    )
    model.summary()
    return model


def load_datasets(partial_load: float = 1.0):
    logging.info("Loading datasets...")
    metadata = load_wikipedia_metadata(WIKIPEDIA_DATA_DIR)
    train_size = int(metadata["train"]["length"] * partial_load)
    logging.info(f"train_size = {train_size}")
    evaluation_size = int(metadata["evaluation"]["length"] * partial_load)
    logging.info(f"evaluation_size = {evaluation_size}")
    test_size = int(metadata["test"]["length"] * partial_load)
    logging.info(f"test_size = {test_size}")
    train_dataset = TextBertAutoencoderWikipediaDataset(
        f"{WIKIPEDIA_DATA_DIR}/train",
        parallel_file_read=NUM_PARALLEL_CALLS,
        batch_size=BATCH_SIZE,
        max_text_length=MAX_TEXT_LENGTH,
        vocabulary=VOCAB_FILE,
        vocabulary_size=VOCAB_SIZE,
    ).take(int(train_size / BATCH_SIZE))
    eval_dataset = TextBertAutoencoderWikipediaDataset(
        f"{WIKIPEDIA_DATA_DIR}/evaluation",
        parallel_file_read=NUM_PARALLEL_CALLS,
        batch_size=BATCH_SIZE,
        max_text_length=MAX_TEXT_LENGTH,
        vocabulary=VOCAB_FILE,
        vocabulary_size=VOCAB_SIZE,
    ).take(int(evaluation_size / BATCH_SIZE))
    test_dataset = TextBertAutoencoderWikipediaDataset(
        f"{WIKIPEDIA_DATA_DIR}/test",
        parallel_file_read=NUM_PARALLEL_CALLS,
        batch_size=BATCH_SIZE,
        max_text_length=MAX_TEXT_LENGTH,
        vocabulary=VOCAB_FILE,
        vocabulary_size=VOCAB_SIZE,
    ).take(int(test_size / BATCH_SIZE))
    logging.info("Datasets loaded.")
    return train_dataset, eval_dataset, test_dataset


def get_checkpoint_dir(model):
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def train_model(model, train_dataset, validation_dataset, test_dataset):
    logging.info("Training model...")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=get_checkpoint_dir(model),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback],
    )
    results = model.evaluate(test_dataset)
    print()
    print(f"Model evaluation: {results}")
    print()
    model.save(f"models/{MODEL_NAME}", overwrite=True)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.INFO)

    logging.info(f"WIKIPEDIA_DATA_DIR = {WIKIPEDIA_DATA_DIR}")
    logging.info(f"WIKIPEDIA_DATASET_SIZE = {WIKIPEDIA_DATASET_SIZE}")
    logging.info(f"MAX_TEXT_LENGTH = {MAX_TEXT_LENGTH}")
    logging.info(f"VOCAB_SIZE = {VOCAB_SIZE}")
    logging.info(f"BATCH_SIZE = {BATCH_SIZE}")
    logging.info(f"EPOCHS = {EPOCHS}")
    logging.info(f"LEARNING_RATE = {LEARNING_RATE}")
    logging.info(f"NUM_PARALLEL_CALLS = {NUM_PARALLEL_CALLS}")
    logging.info(f"DIMENSOES_ESPACO_LATENTE = {DIMENSOES_ESPACO_LATENTE}")
    logging.info(f"MODEL_NAME = {MODEL_NAME}")
    logging.info(f"MODEL_CHECKPOINT = {MODEL_CHECKPOINT}")

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Números de GPUs disponíveis: {gpu_count}")

    train_dataset, eval_dataset, test_dataset = load_datasets(WIKIPEDIA_DATASET_SIZE)
    logging.info(list(train_dataset.take(1)))
    logging.info(train_dataset.element_spec)
    model = create_model()
    train_model(model, train_dataset, eval_dataset, test_dataset)


if __name__ == "__main__":
    main()
