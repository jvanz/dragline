import os
import logging

from transformers import (
    TFBertModel,
    AutoTokenizer,
)
import tensorflow as tf

from gazettes.data import WikipediaDataset

WIKIPEDIA_DATA_DIR = str(os.environ.get("WIKIPEDIA_DATA_DIR", "data/wikipedia"))
WIKIPEDIA_DATASET_SIZE = int(os.environ.get("WIKIPEDIA_DATASET_SIZE", 16450980))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 64))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NUM_PARALLEL_CALLS = tf.data.AUTOTUNE
if "NUM_PARALLEL_CALLS" in os.environ:
    NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS"))
DIMENSOES_ESPACO_LATENTE = int(os.environ.get("DIMENSOES_ESPACO_LATENTE", 32))
DEFAULT_MODEL_NAME = "text_transformer_autoencoder"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT", "neuralmind/bert-base-portuguese-cased"
)
VOCAB_FILE = os.environ.get("VOCAB_FILE", "data/bertimbau_base_vocab.txt")


def load_bertimbau_model():
    return TFBertModel.from_pretrained(MODEL_CHECKPOINT, from_pt=True)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT,
        use_fast=False,
        vocab_file=VOCAB_FILE,
        clean_text=True,
        do_lower_case=True,
    )


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
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["acc"],
    )
    model.summary()
    return model


def get_dataset_partitions(
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


def print_some_records(dataset, num=1):
    for record in dataset.take(num):
        logging.info(repr(record))


def assert_one_hot(dataset):
    for record in dataset.take(1):
        for index, word_index in enumerate(record[0][0]):
            assert record[1][index][word_index] == 1.0


def load_data():
    tokenizer = load_tokenizer()
    dataset = WikipediaDataset(WIKIPEDIA_DATA_DIR)
    print_some_records(dataset)

    def preprocess_text(text):
        tokenizer_output = tokenizer(
            text.numpy().decode("utf8"),
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
        )
        return (
            tokenizer_output["input_ids"],
            tokenizer_output["token_type_ids"],
            tokenizer_output["attention_mask"],
            tokenizer_output["input_ids"],
        )

    def tf_preprocess_text(text):
        preprocessed_text = tf.py_function(
            preprocess_text,
            [text],
            [
                tf.TensorSpec(
                    shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name="input_ids"
                ),
                tf.TensorSpec(
                    shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name="token_type_ids",
                ),
                tf.TensorSpec(
                    shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name="attention_mask",
                ),
                tf.TensorSpec(shape=(MAX_TEXT_LENGTH,), dtype=tf.int32, name="target"),
            ],
        )
        return [tf.reshape(tensor, [MAX_TEXT_LENGTH,]) for tensor in preprocessed_text]

    dataset = dataset.map(
        tf_preprocess_text, num_parallel_calls=NUM_PARALLEL_CALLS, deterministic=False
    )
    print_some_records(dataset)

    def organize_targets(input_ids, token_type_ids, attention_mask, target):
        return (
            (input_ids, token_type_ids, attention_mask),
            tf.one_hot(target, VOCAB_SIZE),
        )

    dataset = dataset.map(organize_targets)
    print_some_records(dataset)
    assert_one_hot(dataset)

    return get_dataset_partitions(dataset, WIKIPEDIA_DATASET_SIZE)


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

    train_dataset, validation_dataset, test_dataset = load_data()
    model = create_model()
    train_model(model, train_dataset, validation_dataset, test_dataset)


if __name__ == "__main__":
    main()
