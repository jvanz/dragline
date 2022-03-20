import os
import logging

import tensorflow as tf

from gazettes.data import WikipediaDataset

WIKIPEDIA_DATA_DIR = os.environ["WIKIPEDIA_DATA_DIR"]
VOCAB_FILE = os.environ["VOCAB_FILE"]
VOCAB_SIZE = int(os.environ["VOCAB_SIZE"])


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(WIKIPEDIA_DATA_DIR)
    logging.info(VOCAB_FILE)
    logging.info(VOCAB_SIZE)

    dataset = WikipediaDataset(f"{WIKIPEDIA_DATA_DIR}/train")

    vectorize_layer = tf.keras.layers.TextVectorization(output_mode="int")

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
