import unittest
import tempfile
import json
import csv

import tensorflow as tf
import numpy as np

from gazettes.data import (
    load_vocabulary_from_tokenizer,
    load_csv_file_column,
    load_pretrained_embeddings,
    load_tokenizer,
    prepare_embedding_matrix,
    get_dataset_stats,
    TextAutoencoderWikipediaCSVDataset,
    START_TOKEN,
    STOP_TOKEN,
    build_vocabulary,
    build_and_save_vocabulary,
    load_vocabulary_from_file,
)


class DataFunctionTestCase(unittest.TestCase):
    def test_load_vocabulary(self):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=True
        ) as tokenizer_file:
            word_counts_config = json.dumps({"eu": 2, "não": 1, "sei": 3})
            json.dump(
                {"config": {"word_counts": word_counts_config}},
                tokenizer_file,
            )
            tokenizer_file.flush()

            vocab_size = 1
            vocabulary = load_vocabulary_from_tokenizer(tokenizer_file.name, vocab_size)
            self.assertEqual(vocabulary, ["", "[UNK]", "sei"])
            vocab_size = 2
            vocabulary = load_vocabulary_from_tokenizer(tokenizer_file.name, vocab_size)
            self.assertEqual(vocabulary, ["", "[UNK]", "sei", "eu"])

    def write_fake_tokenizer_file(self, tokenizer_file):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="[UNK]")
        tokenizer.fit_on_texts(
            [
                "Uma sentença que deve ser utilizada para treinar o tokenizador utilizado para treinar o autoencoder"
            ]
        )
        tokenizer_file.write(tokenizer.to_json())
        tokenizer_file.flush()

    def test_load_tokenizer(self):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=True
        ) as tokenizer_file:
            self.write_fake_tokenizer_file(tokenizer_file)

            tokenizer = load_tokenizer(tokenizer_file.name)
            self.assertEqual(len(tokenizer.word_index), 13)
            self.assertEqual(tokenizer.word_index["[UNK]"], 1)
            self.assertEqual(tokenizer.word_index["para"], 2)

    def write_fake_embedding_file(self, embedding_file):
        embedding_file.write("11 50")
        embedding_file.write("\n")
        embedding_file.write(
            "para -0.865481 -0.941853 -0.137545 -4.104312 -0.564377 -0.121434 0.334428 -0.584465 -0.355633 0.144727 0.877781 -0.202073 0.544712 -0.344749 -0.147748 0.350587 0.125194 0.202245 0.787406 1.387583 -0.139201 -0.178953 -0.495674 -1.202580 0.036768 -0.644099 0.417481 -0.325586 0.374167 1.875644 0.412921 0.040739 0.029373 0.230346 0.533452 -0.186603 0.206326 -0.921190 -1.255070 0.294531 1.077967 -0.852891 -0.439436 -0.039615 0.505775 0.287889 -0.024923 0.137341 0.548348 -0.055328"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "de -0.250167 -0.497073 -0.654335 -4.908071 0.407453 0.262665 0.797045 0.189579 -0.488754 1.189613 1.335908 -0.013967 -0.557421 -0.648254 -0.299232 -0.049164 -0.029339 -1.021244 0.075935 1.147026 1.154523 -0.422338 -0.422481 -1.801476 -0.676745 -0.817729 0.349651 0.198398 -0.008017 0.651300 -0.374300 -0.503768 -0.151164 0.306056 -0.277929 0.749367 -0.243536 0.608048 -0.958676 0.999560 1.227354 0.224327 0.126123 -0.441761 0.989118 0.498052 -0.947346 0.024164 0.153323 0.344427"
        )
        embedding_file.write("\n")
        embedding_file.write(
            ". -0.760922 -0.060111 -0.318281 -3.544602 0.020179 0.242053 0.430245 -0.660070 0.016198 0.819403 0.647113 -0.103079 -0.145391 0.237824 -0.709342 -0.505342 1.070425 0.100801 0.468127 1.194871 1.161511 0.463369 -0.256357 -1.569226 -1.126657 -0.660804 -0.385289 -1.043691 -0.179519 2.488872 0.001941 -0.033683 -0.567296 -0.388845 -0.309537 -0.557241 0.363696 -0.273223 -0.724113 1.162116 1.842170 -0.420239 -0.456461 -0.729635 0.701809 0.703842 0.128216 0.279015 0.423745 -0.515101"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "a -0.896791 -0.933029 0.172109 -4.839827 0.550315 0.187177 -0.225299 -0.600800 -0.335084 0.718750 0.572709 -0.591387 0.392771 0.138562 0.224216 -0.332028 0.326385 -1.232845 0.404873 1.066597 0.921141 0.120547 0.229917 -1.374687 -0.178624 0.671497 1.074445 -0.353192 -0.514254 1.101591 -0.395287 0.281086 0.346556 -0.429881 -0.517540 -0.424209 0.882984 -0.374556 -1.196126 0.286940 0.114625 -0.151498 0.088812 0.048362 -0.039576 -0.491554 0.579065 -0.407343 -0.345218 -0.077220"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "o -0.632575 -0.115801 0.553089 -4.183989 0.425046 1.715943 -0.027738 -0.431429 -0.891655 1.007702 0.888476 -0.439783 -0.281935 -0.471151 -0.022360 0.458956 0.537205 -0.103836 0.651895 0.418140 0.299266 0.313891 -0.724487 -1.158407 -1.182278 0.015967 1.192445 -0.737702 0.277540 0.979173 0.548221 0.455641 0.154486 0.222951 1.622088 -1.477954 -0.105596 0.245025 -1.674061 0.341556 0.406670 0.045055 -0.717476 -0.272441 0.243455 1.233958 0.390470 0.350278 0.277538 -0.110385"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "e -0.374438 -0.536315 -0.758331 -3.704208 0.226590 -0.010066 0.347838 -0.706328 -0.454332 0.525428 1.271844 -0.747334 0.054796 -0.400851 0.339300 -0.326526 0.559682 -0.335690 0.913415 0.991717 -0.069873 0.356918 -0.923631 -1.518439 -0.658405 -1.404329 0.470226 -0.399750 -0.218981 1.724632 0.319832 1.021842 0.551652 0.347276 -0.221634 0.338634 -0.210083 -1.071705 -1.150460 0.373328 1.460433 -0.272520 -0.638388 0.531466 0.477762 -0.186777 0.296675 -0.347883 -0.133917 -0.087929"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "que -0.914492 -0.583486 -0.710203 -3.806935 0.146957 0.524191 -0.507795 -0.202060 -0.892379 1.332948 0.461956 0.021763 0.097482 -0.458620 -0.573615 0.295283 0.270094 -0.649586 0.931590 1.586553 1.420690 0.634359 -0.029634 -1.268311 -0.083590 -0.126941 0.739827 -1.193557 0.068180 1.272291 -0.438969 0.628654 -0.283140 0.288809 0.078296 -1.078349 0.280183 -0.257799 -1.133783 0.028817 -0.069857 -0.630516 -0.459928 0.419947 -0.032111 0.786083 -0.029576 0.684914 0.467038 -0.386656"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "do -0.371905 1.662570 0.762216 -4.192826 0.204264 1.410739 0.065396 -0.707296 -0.694268 0.789545 1.831975 -0.256575 -0.793970 -0.792104 -0.459066 0.082525 0.726479 -0.625122 -0.301380 0.648220 0.411911 0.063700 -0.356437 -1.381659 -0.444857 0.366380 0.207816 -0.374976 0.519222 1.040977 0.564709 0.057061 0.247889 0.109462 1.122942 -1.060629 -0.486961 -0.302114 -1.532192 0.876942 0.604553 0.526120 -0.614471 -0.831563 0.562891 0.269603 -0.026146 0.133503 -0.396476 0.117474"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "da -0.614247 1.132003 0.498888 -4.771135 0.284453 0.211812 -0.123134 -0.990161 -0.392569 0.695900 1.424825 -0.355684 -0.082200 -0.280058 0.672518 -0.177000 0.902485 -0.885832 -0.262943 0.961541 0.323212 0.504100 -0.051790 -1.148939 0.644860 0.902511 0.364905 -0.829119 -0.594093 1.290525 -0.322842 -0.215406 -0.647396 -0.627145 -0.528855 -0.336877 -0.212425 -1.074966 -1.085090 0.518299 0.530463 0.426514 -0.143098 -0.679470 0.131698 -0.729273 -0.021690 -0.048997 -0.264199 -0.009548"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "treinar -0.409770 -1.233237 -0.603106 -4.115523 0.068511 0.740358 -0.198917 -0.568800 0.082770 1.510337 1.050147 -0.442545 -0.491040 -0.226286 -0.373659 -0.053119 -0.109965 -0.166632 1.363881 0.489029 0.817113 -0.946224 -0.188450 -1.674871 -0.303241 -0.233369 0.262946 0.152489 0.334662 1.674697 0.487663 -0.709112 -0.527339 -0.331422 -0.187952 0.062751 0.252876 -0.141749 -1.359539 0.484522 0.427300 -0.150440 -0.256369 0.042664 0.726409 -0.045237 0.567325 0.451937 0.431785 -0.120121"
        )
        embedding_file.write("\n")
        embedding_file.write(
            "<unk> 0.155503 0.112883 -0.072014 0.298080 -0.015806 -0.028018 -0.107673 0.026280 0.037245 -0.205769 -0.158379 0.074089 -0.015952 0.095512 0.124618 0.000019 -0.083389 0.105941 -0.126176 -0.207609 0.004121 -0.009182 0.173573 0.216759 0.132383 0.005677 -0.088592 0.007655 0.023474 -0.176095 0.100153 0.027912 0.032341 0.073515 -0.051644 0.116569 0.059340 0.105374 0.127422 -0.143274 -0.183256 0.050080 0.008661 -0.067929 -0.064259 -0.038466 0.011080 -0.091202 0.094777 0.015651"
        )
        embedding_file.write("\n")
        embedding_file.flush()

    def test_load_embeddings(self):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=True
        ) as embedding_file:
            self.write_fake_embedding_file(embedding_file)

            embeddings = load_pretrained_embeddings(embedding_file.name)
            self.assertEqual(len(embeddings), 11)

    def test_prepare_embedding_matrix(self):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=True
        ) as embedding_file:
            self.write_fake_embedding_file(embedding_file)
            embeddings = load_pretrained_embeddings(embedding_file.name)

            with tempfile.NamedTemporaryFile(
                "w", encoding="utf8", delete=True
            ) as tokenizer_file:
                self.write_fake_tokenizer_file(tokenizer_file)

                embedding_matrix = prepare_embedding_matrix(
                    tokenizer_file.name, embedding_file.name, 2
                )
                self.assertEqual(embedding_matrix.shape, (4, embeddings.vector_size))
                self.assertTrue(
                    np.array_equal(
                        embedding_matrix[0], np.zeros((embeddings.vector_size))
                    )
                )
                self.assertTrue(
                    np.array_equal(embedding_matrix[1], embeddings.get_vector("<unk>"))
                )
                self.assertTrue(
                    np.array_equal(embedding_matrix[2], embeddings.get_vector("para"))
                )
                self.assertTrue(
                    np.array_equal(
                        embedding_matrix[3], embeddings.get_vector("treinar")
                    )
                )

    def test_load_csv_file_collumn(self):
        csv_file_name = None
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=False
        ) as csv_file:
            fieldnames = ["text"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            writer.writerow({"text": "first text"})
            writer.writerow({"text": "second text"})
            csv_file.flush()
            csv_file_name = csv_file.name
        entries = list(load_csv_file_column(csv_file_name, "text"))
        self.assertEqual(entries, ["first text", "second text"])

    def test_dataset_stats(self):
        stats = get_dataset_stats(
            [
                ["This", "is", "a", "sentence"],
                ["This", "is", "a", "second", "sentence"],
                ["I", "do", "not", "know", "what", "to", "write", "there"],
            ]
        )
        self.assertEqual(
            stats,
            {
                "max_sequence_length": 8,
                "min_sequence_length": 4,
                "average_sequence_length": 6,
            },
        )

    def test_build_vocabulary(self):
        dataset = [
            "Essa é uma sentença do dataset",
            "Essa é outra sentença do dataset",
            "Qualuer outra frase que possa ter alguns tokens repetidos",
        ]
        expected_vocabulary = [
            "é",
            "sentença",
            "outra",
            "essa",
            "do",
            "dataset",
            "uma",
            "tokens",
            "ter",
            "repetidos",
            "que",
            "qualuer",
            "possa",
            "frase",
            "alguns",
        ]

        def get_dataset():
            for sample in dataset:
                yield sample

        vocabulary = build_vocabulary(
            tf.data.Dataset.from_generator(
                get_dataset, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string))
            )
        )
        self.assertEqual(vocabulary, expected_vocabulary)

    def test_build_and_save_vocabulary(self):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=True
        ) as vocab_file:

            dataset = [
                "Essa é uma sentença do dataset",
                "Essa é outra sentença do dataset",
                "Qualuer outra frase que possa ter alguns tokens repetidos",
            ]
            expected_vocabulary = [
                "é",
                "sentença",
                "outra",
                "essa",
                "do",
                "dataset",
                "uma",
                "tokens",
                "ter",
                "repetidos",
                "que",
                "qualuer",
                "possa",
                "frase",
                "alguns",
            ]

            def get_dataset():
                for sample in dataset:
                    yield sample

            build_and_save_vocabulary(
                tf.data.Dataset.from_generator(
                    get_dataset,
                    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
                ),
                vocab_file.name,
            )
            with open(vocab_file.name, "r") as check_vocab_file:
                vocab = []
                for line in check_vocab_file:
                    vocab.append(line.strip())
                self.assertEqual(expected_vocabulary, vocab)

    def test_load_vocab_from_file(self):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=True
        ) as vocab_file:
            expected_vocabulary = [
                "é",
                "sentença",
                "outra",
                "essa",
                "do",
                "dataset",
                "uma",
                "tokens",
                "ter",
                "repetidos",
                "que",
                "qualuer",
                "possa",
                "frase",
                "alguns",
            ]
            for token in expected_vocabulary:
                vocab_file.write(token)
                vocab_file.write("\n")
            vocab_file.flush()
            vocabulary = load_vocabulary_from_file(vocab_file.name)
            self.assertEqual(expected_vocabulary, vocabulary)

            vocab_size = 2
            vocabulary = load_vocabulary_from_file(
                vocab_file.name, vocabulary_size=vocab_size
            )
            self.assertEqual(expected_vocabulary[:2], vocabulary)


class TextAutoencoderWikipediaDatasetCSVTests(unittest.TestCase):
    def write_csv_file(self, csv_file, data):
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for column in data:
            for row in data[column]:
                writer.writerow({column: row})
        csv_file.flush()

    def test_load_csv_file_into_dataset(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf8", delete=True) as csv_file:
            data = {"text": ["This is the first row", "This is the second row"]}
            self.write_csv_file(csv_file, data)
            dataset = TextAutoencoderWikipediaCSVDataset(
                csv_file.name, deterministic=True
            )
            self.assertIsNotNone(dataset)
            data = list(dataset.as_numpy_iterator())
            self.assertEqual(
                data,
                [
                    (b"This is the first row", b"This is the first row"),
                    (b"This is the second row", b"This is the second row"),
                ],
            )

    def test_add_start_stop_token_dataset(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf8", delete=True) as csv_file:
            data = {"text": ["This is the first row", "This is the second row"]}
            self.write_csv_file(csv_file, data)
            dataset = TextAutoencoderWikipediaCSVDataset(
                csv_file.name,
                start_token=START_TOKEN,
                stop_token=STOP_TOKEN,
                deterministic=True,
            )

            self.assertIsNotNone(dataset)
            data = list(dataset.as_numpy_iterator())
            self.assertEqual(
                data,
                [
                    (
                        b"This is the first row",
                        f"{START_TOKEN} This is the first row {STOP_TOKEN}".encode(
                            "utf8"
                        ),
                    ),
                    (
                        b"This is the second row",
                        f"{START_TOKEN} This is the second row {STOP_TOKEN}".encode(
                            "utf8"
                        ),
                    ),
                ],
            )

    def test_text_vectorization(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf8", delete=True) as csv_file:
            dataset = ["This is the first row", "This is the second row"]
            data = {"text": dataset}
            self.write_csv_file(csv_file, data)

            def get_dataset():
                for sample in dataset:
                    yield sample

            text_vectorization = tf.keras.layers.TextVectorization()
            text_vectorization.adapt(
                tf.data.Dataset.from_generator(
                    get_dataset,
                    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
                )
            )

            dataset = TextAutoencoderWikipediaCSVDataset(
                csv_file.name,
                start_token=START_TOKEN,
                stop_token=STOP_TOKEN,
                text_vectorization=text_vectorization,
                deterministic=True,
                add_decoder_input=True,
            )

            self.assertIsNotNone(dataset)
            data = list(dataset.as_numpy_iterator())
            expected_output = [
                (
                    (
                        text_vectorization(
                            tf.constant(f"This is the first row")
                        ).numpy(),
                        text_vectorization(
                            tf.constant(
                                f"{START_TOKEN} This is the first row {STOP_TOKEN}"
                            )
                        ).numpy(),
                    ),
                    text_vectorization(
                        tf.constant(f"{START_TOKEN} This is the first row {STOP_TOKEN}")
                    ).numpy(),
                ),
                (
                    (
                        text_vectorization(
                            tf.constant(f"This is the second row")
                        ).numpy(),
                        text_vectorization(
                            tf.constant(
                                f"{START_TOKEN} This is the second row {STOP_TOKEN}"
                            )
                        ).numpy(),
                    ),
                    text_vectorization(
                        tf.constant(
                            f"{START_TOKEN} This is the second row {STOP_TOKEN}"
                        )
                    ).numpy(),
                ),
            ]
            for output, expected in zip(data, expected_output):
                self.assertTrue(np.array_equal(output[0][0], expected[0][0]))
                self.assertTrue(np.array_equal(output[0][1], expected[0][1]))
                self.assertTrue(np.array_equal(output[1], expected[1]))

    def test_text_vectorization_max_text_length(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf8", delete=True) as csv_file:
            dataset = ["This is the first row"]
            data = {"text": dataset}
            self.write_csv_file(csv_file, data)

            def get_dataset():
                for sample in dataset:
                    yield sample

            text_vectorization = tf.keras.layers.TextVectorization(
                output_sequence_length=10,
            )
            text_vectorization.adapt(
                tf.data.Dataset.from_generator(
                    get_dataset,
                    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
                )
            )

            dataset = TextAutoencoderWikipediaCSVDataset(
                csv_file.name,
                start_token=START_TOKEN,
                stop_token=STOP_TOKEN,
                text_vectorization=text_vectorization,
                deterministic=True,
            )

            self.assertIsNotNone(dataset)
            data = list(dataset.as_numpy_iterator())
            expected_output = [
                (
                    np.array([2, 5, 3, 6, 4, 0, 0, 0, 0, 0]),
                    np.array([1, 2, 5, 3, 6, 4, 1, 0, 0, 0]),
                ),
            ]

            print(data)
            print(expected_output)
            for output, expected in zip(data, expected_output):
                self.assertTrue(np.array_equal(output[0][0], expected[0][0]))
                self.assertTrue(np.array_equal(output[0][1], expected[0][1]))

    def test_text_one_hot(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf8", delete=True) as csv_file:
            dataset = ["This is the first row"]
            data = {"text": dataset}
            self.write_csv_file(csv_file, data)

            def get_dataset():
                for sample in dataset:
                    yield sample

            text_vectorization = tf.keras.layers.TextVectorization()
            text_vectorization.adapt(
                tf.data.Dataset.from_generator(
                    get_dataset,
                    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
                )
            )

            dataset = TextAutoencoderWikipediaCSVDataset(
                csv_file.name,
                start_token=START_TOKEN,
                stop_token=STOP_TOKEN,
                text_vectorization=text_vectorization,
                deterministic=True,
                add_decoder_input=False,
                one_hot=True,
                vocabulary_size=7,
            )

            self.assertIsNotNone(dataset)
            data = list(dataset.as_numpy_iterator())
            expected_output = [
                (
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                )
            ]
            # Jkprint(data)
            # Jkprint(expected_output)
            # Jkfor output, expected in zip(data, expected_output):
            # Jk    self.assertTrue(np.array_equal(output[0], expected[0]))
            # Jk    self.assertTrue(np.array_equal(output[1], expected[1]))


class TensorflowTests(unittest.TestCase):
    def test_text_vectorization(self):
        dataset = [
            "STX Essa é uma sentença do dataset ETX",
            "STX Essa é outra sentença do dataset ETX",
            "STX Qualquer outra frase que possa ter alguns tokens repetidos ETX",
        ]

        def get_dataset():
            for sample in dataset:
                yield sample

        text_vectorization = tf.keras.layers.TextVectorization()
        text_vectorization.adapt(
            tf.data.Dataset.from_generator(
                get_dataset, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string))
            )
        )
        text_vectorization2 = tf.keras.layers.TextVectorization(
            vocabulary=text_vectorization.get_vocabulary()
        )

        sentence = "Uma frase qualquer para tester vectorizadores"
        sentence_vec1 = text_vectorization(tf.constant(sentence)).numpy()
        sentence_vec2 = text_vectorization2(tf.constant(sentence)).numpy()
        self.assertTrue(np.array_equal(sentence_vec1, sentence_vec2))

        self.assertEqual(
            text_vectorization.get_vocabulary(False),
            text_vectorization2.get_vocabulary(False),
        )
        self.assertEqual(
            text_vectorization.get_vocabulary(), text_vectorization2.get_vocabulary()
        )
        self.assertEqual("", text_vectorization2.get_vocabulary()[0])
        self.assertEqual("[UNK]", text_vectorization2.get_vocabulary()[1])

    def test_model_with_text_vectorization(self):
        dataset = [
            "STX Essa é uma sentença do dataset ETX",
            "STX Essa é outra sentença do dataset ETX",
            "STX Qualquer outra frase que possa ter alguns tokens repetidos ETX",
        ]

        def get_dataset():
            for sample in dataset:
                yield sample

        text_vectorization = tf.keras.layers.TextVectorization()
        text_vectorization.adapt(
            tf.data.Dataset.from_generator(
                get_dataset, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string))
            )
        )

        model = tf.keras.Sequential()
        model.add(text_vectorization)
        output = model.predict(["Uma frase qualquer para tester vectorizadores"])
        expected_output = np.array([[10, 17, 15, 1, 1, 1]])
        self.assertTrue(np.array_equal(output, expected_output))

    def test_model_with_string_input_and_text_vectorization(self):
        dataset = [
            "STX Essa é uma sentença do dataset ETX",
            "STX Essa é outra sentença do dataset ETX",
            "STX Qualquer outra frase que possa ter alguns tokens repetidos ETX",
        ]

        def get_dataset():
            for sample in dataset:
                yield sample

        text_vectorization = tf.keras.layers.TextVectorization(
            output_sequence_length=10, pad_to_max_tokens=True, max_tokens=20
        )
        text_vectorization.adapt(
            tf.data.Dataset.from_generator(
                get_dataset, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string))
            )
        )

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(None,), dtype=tf.string))
        model.add(text_vectorization)
        output = model.predict(["Uma frase qualquer para tester vectorizadores"])
        expected_output = np.array([[10, 17, 15, 1, 1, 1, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(output, expected_output))

    def test_model_with_one_hot_lambda_layer(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Lambda(tf.one_hot, arguments={"depth": 3}))
        output = model.predict([0, 1, 2])
        expected_output = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.array_equal(output, expected_output))
