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

PADDING_TOKEN = "<PAD>"
UNK_TOKEN = "[UNK]"


def create_checkpoint_dir(checkpoint_dir, model_name):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(str(checkpoint_dir) + f"/{model_name}", exist_ok=True)


def create_model(
    embedding_matrix,
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
    vocab_size,
):
    logging.info("Creating model...")

    encoder_input = tf.keras.layers.Input(shape=(max_text_length,), dtype="int64")
    embedding = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dimensions,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )(encoder_input)
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
        encoder = tf.keras.layers.Bidirectional(layer, merge_mode="sum")(embedding)
    else:
        encoder = layer(embedding)

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
    # metrics = [tf.keras.metrics.MeanSquaredError()]

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
    embedding_matrix,
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
    vocab_size,
    from_scratch,
    checkpoint_dir,
):
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir, model_name)
    if from_scratch or latest_checkpoint is None:
        print("Creating a new model")
        model = create_model(
            embedding_matrix,
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
            vocab_size,
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
        + "epoch.{epoch:04d}",
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    model_weights_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir)
        + f"/{model_name}/"
        + "weights.{epoch:02d}-{loss:.6f}",
        save_weights_only=False,
        monitor="loss",
        mode="min",
        save_best_only=True,
        save_freq=patience,
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
            # model_checkpoint_callback,
            model_weights_checkpoint_callback,
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
                embedding_sentence.append(embeddingmodel.get_vector(UNK_TOKEN))
        if len(embedding_sentence) > max_text_length:
            embedding_sentence = embedding_sentence[:max_text_length]
        if len(embedding_sentence) < max_text_length:
            for _ in range(max_text_length - len(embedding_sentence)):
                embedding_sentence.append(embeddingmodel.get_vector(PADDING_TOKEN))
        assert len(embedding_sentence) == max_text_length
        yield embedding_sentence


def convert_embedding_sentence_to_string_sentence(sentence):
    strsentence = []
    for emb in sentence:
        word = embeddingmodel.similar_by_vector(emb)[0][0]
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
        .prefetch(8)
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
        .prefetch(8)
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
        .prefetch(8)
    )
    return train_dataset, eval_dataset, test_dataset


def load_dataset(
    dataset_dir: str, batch_size, max_text_length, embedding_dim, num_parallel_calls
):
    logging.info("Loading datasets...")
    metadata = load_wikipedia_metadata(dataset_dir)

    train_dataset = WikipediaDataset(
        f"{dataset_dir}/train", batch_size=batch_size
    ).prefetch(batch_size)
    eval_dataset = WikipediaDataset(
        f"{dataset_dir}/evaluation", batch_size=batch_size
    ).prefetch(batch_size)
    test_dataset = WikipediaDataset(
        f"{dataset_dir}/test", batch_size=batch_size
    ).prefetch(batch_size)
    return train_dataset, eval_dataset, test_dataset


def compare_original_and_generated_sentences(inputs, predictions):
    for inputt, prediction in zip(inputs.unbatch().as_numpy_iterator(), predictions):
        inputt = convert_embedding_sentence_to_string_sentence(inputt)
        prediction = convert_embedding_sentence_to_string_sentence(prediction)
        logging.info(f"{inputt} -> {prediction}")


def evaluate_model(dataset, model_path):
    logging.info(f"Loading model {model_path}")
    model = tf.keras.models.load_model(model_path, compile=True)
    model.summary()
    results = model.evaluate(dataset, return_dict=True,)
    print()
    print(f"Loaded model evaluation: {results}")
    print()


def predict_text(dataset, model_path, vectorization_layer):
    logging.info(f"Loading model {model_path}")
    model = tf.keras.models.load_model(model_path, compile=True)
    model.summary()
    dataset = dataset.map(
        lambda inputt, target: inputt,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    inputt = list(dataset.unbatch().take(1))[0]
    print(inputt)
    strs = [vectorization_layer.get_vocabulary()[word] for word in inputt]
    print(" ".join(strs))

    inputt = list(dataset.unbatch().batch(1).take(1))
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
        "--train", required=False, action="store_true", help="Train model from scratch",
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
        "--bidirectional-hidden-layers", required=False, action="store_true", help="",
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


def prepare_vectorization_layer(train_dataset, vocab_size, max_text_length):
    logging.info("Vectorizing dataset...")
    vectorization_layer = tf.keras.layers.TextVectorization(
        vocab_size, output_sequence_length=max_text_length, pad_to_max_tokens=True
    )
    vectorization_layer.adapt(train_dataset)
    return vectorization_layer


def get_word_index(vectorization_layer):
    voc = vectorization_layer.get_vocabulary()
    return dict(zip(voc, range(len(voc))))


def generate_embedding_matrix(word_index, vocab_size, embedding_dimensions):
    logging.info("Building embedding matrix")
    embedding_matrix = np.zeros((vocab_size, embedding_dimensions))
    hits = 0
    misses = 0

    for word, i in word_index.items():
        if i == 0:
            embedding_matrix[i] = embeddingmodel.get_vector(PADDING_TOKEN)
        elif i == 1:
            embedding_matrix[i] = embeddingmodel.get_vector(UNK_TOKEN)
        elif word in embeddingmodel:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embeddingmodel.get_vector(word)
            hits += 1
        else:
            misses += 1
    logging.info("Converted %d words (%d misses)" % (hits, misses))
    logging.info(f"Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix


def vectorize_and_add_target_dataset(
    dataset,
    max_text_length,
    vectorization_layer,
    embedding_matrix,
    batch_size,
    embedding_dimensions,
):
    def vectorize_sentence(sentence):
        sentence = vectorization_layer(sentence)
        embsentence = []
        for word in sentence:
            embsentence.append(embedding_matrix[word])
        return sentence, embsentence

    def tf_vectorize_sentence(text):
        return tf.py_function(
            vectorize_sentence,
            [text],
            [
                tf.TensorSpec(shape=(max_text_length,), dtype=tf.int64),
                tf.TensorSpec(
                    shape=(max_text_length, embedding_dimensions,), dtype=tf.float32
                ),
            ],
        )

    return (
        dataset.unbatch()
        .map(
            tf_vectorize_sentence,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .batch(batch_size)
    )


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
        PADDING_TOKEN, np.random.uniform(-1, 1, args.embedding_dimensions)
    )
    embeddingmodel.add_vector(
        UNK_TOKEN, np.random.uniform(-1, 1, args.embedding_dimensions)
    )
    # add 2 to cover unk and pad tokens
    args.vocab_size += 2

    gpu_count = len(tf.config.list_physical_devices("GPU"))
    logging.info(f"Números de GPUs disponíveis: {gpu_count}")

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

    vectorization_layer = prepare_vectorization_layer(
        train_dataset, args.vocab_size, args.max_text_length
    )
    word_index = get_word_index(vectorization_layer)
    logging.info(f"Word index size: {len(word_index)}")
    embedding_matrix = generate_embedding_matrix(
        word_index, args.vocab_size, args.embedding_dimensions
    )

    train_dataset = vectorize_and_add_target_dataset(
        train_dataset,
        args.max_text_length,
        vectorization_layer,
        embedding_matrix,
        args.batch_size,
        args.embedding_dimensions,
    )
    eval_dataset = vectorize_and_add_target_dataset(
        eval_dataset,
        args.max_text_length,
        vectorization_layer,
        embedding_matrix,
        args.batch_size,
        args.embedding_dimensions,
    )
    test_dataset = vectorize_and_add_target_dataset(
        test_dataset,
        args.max_text_length,
        vectorization_layer,
        embedding_matrix,
        args.batch_size,
        args.embedding_dimensions,
    )
    logging.info(train_dataset.element_spec)
    logging.info(eval_dataset.element_spec)
    logging.info(test_dataset.element_spec)

    # sentences = list(train_dataset.unbatch().take(5))
    # for inputt, output in sentences:
    #    strs = [vectorization_layer.get_vocabulary()[word] for word in inputt]
    #    print(" ".join(strs))
    #    emnsentence = []
    #    for i, emb in enumerate(output.numpy()):
    #        assert np.array_equal(emb, embedding_matrix[inputt[i]])
    #        matches = embeddingmodel.similar_by_vector(emb)
    #        emnsentence.append(matches[0][0])
    #    print(" ".join(emnsentence))

    if args.train:
        model = create_or_load_model(
            embedding_matrix,
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
            args.vocab_size,
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
        predict_text(test_dataset, args.save_model_at, vectorization_layer)


if __name__ == "__main__":
    main()
