import os
import logging

import tensorflow as tf

from gazettes.data import WikipediaDataset

WIKIPEDIA_DATA_DIR = os.environ["WIKIPEDIA_DATA_DIR"]
VOCAB_FILE = os.environ["VOCAB_FILE"]
VOCAB_SIZE = None
if "VOCAB_SIZE" in os.environ and len(os.environ["VOCAB_SIZE"]) > 0:
    VOCAB_SIZE = int(os.environ["VOCAB_SIZE"])


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"WIKIPEDIA_DATA_DIR: {WIKIPEDIA_DATA_DIR}")
    logging.info(f"VOCAB_FILE: {VOCAB_FILE}")
    logging.info(f"VOCAB_SIZE: {VOCAB_SIZE}")

    dataset = WikipediaDataset(f"{WIKIPEDIA_DATA_DIR}/train")

    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode="int", max_tokens=VOCAB_SIZE
    )

    logging.info(f"Adapting TextVectorization layers")
    vectorize_layer.adapt(dataset)

    # save vocabulary
    logging.info(f"Vocabulary size: {vectorize_layer.vocabulary_size()}")

    with open(VOCAB_FILE, "w") as vocab:
        for token in vectorize_layer.get_vocabulary(include_special_tokens=False):
            vocab.write(token)
            vocab.write("\n")


if __name__ == "__main__":
    main()
