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
    TextAutoencoderWikipediaCSVDataset,
    START_TOKEN,
    STOP_TOKEN,
)
from gazettes.data import (
    load_pretrained_embeddings,
    load_tokenizer,
    prepare_embedding_matrix,
    load_vocabulary_from_file,
)


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
    vocabulary,
):
    logging.info("Creating model...")

    encoder_input = tf.keras.Input(
        shape=(
            max_text_length,
            len(vocabulary),
        )
    )
    encoder_layers_arguments = {
        "name": "encoder",
        "units": dimensoes_espaco_latent,
        "dropout": dropout,
        "activation": activation,
        "return_state": True,
    }
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
        **encoder_layers_arguments
    )(encoder_input)
    encoder_states = (state_h, state_c)

    decoder_input = tf.keras.Input(
        shape=(
            max_text_length,
            len(vocabulary),
        )
    )
    decoder_layer_arguments = {
        "name": "decoder",
        "units": dimensoes_espaco_latent,
        "return_sequences": True,
        "dropout": dropout,
        "activation": activation,
        "return_state": True,
    }
    decoder_output, _, _ = tf.keras.layers.LSTM(**decoder_layer_arguments)(
        decoder_input, initial_state=encoder_states
    )
    decoder_output = tf.keras.layers.Dense(len(vocabulary), activation="softmax")(
        decoder_output
    )

    model = tf.keras.Model(
        inputs=[encoder_input, decoder_input], outputs=decoder_output, name=model_name
    )

    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    metrics = [tf.keras.metrics.CategoricalCrossentropy(), "accuracy"]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
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
    vocabulary,
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
            vocabulary,
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
    tb_callback = tf.keras.callbacks.TensorBoard(
        "./logs", update_freq=1000, profile_batch="10, 20"
    )
    nan_callback = tf.keras.callbacks.TerminateOnNaN()

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            model_weights_checkpoint_callback,
            early_stop_callback,
            # tb_callback,
            nan_callback,
        ],
    )


def load_dataset(dataset_dir: str, batch_size, text_vectorization, vocabulary_size):
    logging.info("Loading datasets...")

    train_dataset = TextAutoencoderWikipediaCSVDataset(
        f"{dataset_dir}/train.csv",
        start_token=START_TOKEN,
        stop_token=STOP_TOKEN,
        add_decoder_input=True,
        text_vectorization=text_vectorization,
        one_hot=True,
        vocabulary_size=vocabulary_size,
    ).batch(batch_size)
    eval_dataset = TextAutoencoderWikipediaCSVDataset(
        f"{dataset_dir}/eval.csv",
        start_token=START_TOKEN,
        stop_token=STOP_TOKEN,
        add_decoder_input=True,
        text_vectorization=text_vectorization,
        one_hot=True,
        vocabulary_size=vocabulary_size,
    ).batch(batch_size)
    test_dataset = TextAutoencoderWikipediaCSVDataset(
        f"{dataset_dir}/test.csv",
        start_token=START_TOKEN,
        stop_token=STOP_TOKEN,
        add_decoder_input=True,
        text_vectorization=text_vectorization,
        one_hot=True,
        vocabulary_size=vocabulary_size,
    ).batch(batch_size)
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
        "--tokenizer-config-file",
        required=True,
        type=pathlib.Path,
        help="This file is not used to build a tokenizer. It is used to get the vocabulary only",
    )
    parser.add_argument(
        "--vocabulary-file",
        required=True,
        type=pathlib.Path,
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
    parser.add_argument("--max-text-length", required=False, type=int, default=20)
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

    vocabulary = load_vocabulary_from_file(
        args.vocabulary_file, vocabulary_size=args.vocab_size
    )
    text_vectorization = tf.keras.layers.TextVectorization(
        vocabulary=vocabulary, output_sequence_length=args.max_text_length
    )
    logging.debug(text_vectorization.get_vocabulary()[:10])

    train_dataset, eval_dataset, test_dataset = load_dataset(
        args.dataset_dir, args.batch_size, text_vectorization, len(vocabulary)
    )
    logging.info(train_dataset.element_spec)

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
            vocabulary,
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

    if False and args.evaluate:
        # TODO rebuild the model to be able to predict text
        evaluate_model(test_dataset, args.save_model_at)


if __name__ == "__main__":
    main()
