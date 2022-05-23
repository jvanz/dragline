import glob
import os

from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data

import numpy as np
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, SimpleRNN

word_vectors = get_data("wv")


def pre_process_data(filepath):
    positive_path = os.path.join(filepath, "pos")
    negative_path = os.path.join(filepath, "neg")
    pos_label = 1
    neg_label = 0
    dataset = []
    for filename in glob.glob(os.path.join(positive_path, "*.txt")):
        with open(filename, "r") as f:
            dataset.append((pos_label, r.read()))
    for filename in glob.glob(os.path.join(negative_path, "*.txt")):
        with open(filename, "r") as f:
            dataset.append((neg_label, r.read()))
    shuffle(dataset)
    return dataset


def tokenize_and_vectoriza(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        semple_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data


def collect_expected(dataset):
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


dataset = pre_process_data("./aclimdb/train")
vectorized_data = tokenize_and_vectoriza(dataset)
expected = collect_expected(dataset)
split_point = int(len(vectorized_data) * 0.8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

maxlen = 400
batch_size = 32
embedding_dim = 300
epochs = 2

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)


x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dim))
y_train = np.array(y_train)

x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dim))
y_test = np.array(y_test)

num_neurons = 50
model = Sequential()
model.add(
    SimpleRNN(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dim))
)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile("rmsprop", "binary_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
)
model_structure = model.to_json()
with open("simplernn_model1.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("simplernn_model1.h5")
