import unittest
import tempfile
import string
import json
import csv
import os

import numpy as np

try:
    import datasets
    from datasets import load_dataset
    from transformers import BertTokenizer
except Exception:
    print("cannot import datasets")


def missing_datasets():
    try:
        datasets.__version__
    except:
        return True
    return False


def valid_string(string_value):
    translate_table = str.maketrans("", "", string.punctuation)
    return len(string_value.translate(translate_table).strip()) > 0


@unittest.skipIf(missing_datasets(), "No Datasets lib detected. Skipping test.")
class HuggingFacesDatasetTests(unittest.TestCase):
    def write_csv_file(self, data):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=False
        ) as csv_file:
            data = {"text": data}
            fieldnames = list(data.keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for column in data:
                for row in data[column]:
                    writer.writerow({column: row})
            csv_file.flush()
            return csv_file.name

    model_checkpoint = "neuralmind/bert-base-portuguese-cased"

    def test_load_csv_file_into_dataset(self):
        data = ["This is the first row", "This is the second row"]
        csv_file = self.write_csv_file(data)
        dataset = load_dataset("csv", data_files=csv_file)
        self.assertEqual(dataset["train"].num_rows, 2)

    def test_tokenize_csv_file_dataset(self):
        data = ["Essa é a primeira frase", "Essa é a segunda linha"]
        csv_file = self.write_csv_file(data)
        tokenizer = BertTokenizer.from_pretrained(
            self.model_checkpoint,
            do_lower_case=False,
            use_fast=False,
            bos_token_id=101,
            eos_token_id=102,
        )

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = load_dataset("csv", data_files=csv_file)
        dataset = dataset.map(tokenize_function, batched=True)
        self.assertEqual(dataset["train"].num_rows, 2)
        self.assertEqual(
            dataset["train"].column_names,
            ["text", "input_ids", "token_type_ids", "attention_mask"],
        )

    def test_invalid_text(self):
        data = ["This is the first row", ", , , ,", "   "]
        csv_file = self.write_csv_file(data)
        dataset = load_dataset("csv", data_files=csv_file)
        self.assertEqual(dataset["train"].num_rows, 2)

        def invalid_string(examples):
            return [valid_string(example) for example in examples["text"]]

        dataset = dataset.filter(invalid_string, batched=True)
        self.assertEqual(dataset["train"].num_rows, 1)

        data = [
            "Comunas da Alta Saboia",
            "As garantias de TCP envolvem retransmissão e espera de dados, como consequência, intensificam os efeitos de uma alta latência de rede",
            "Dois anos depois, a formação passa a ter licença belga",
            "Savola inen participou de seis edições dos Jogos Olímpicos, de 1928 a 1952",
            "A delegação tinha o objetivo de conquistar 19 medalhas e ficar entre os 12 mais nesse critério",
        ]

        csv_file = self.write_csv_file(data)
        dataset = load_dataset("csv", data_files=csv_file)
        self.assertEqual(dataset["train"].num_rows, len(data))
        dataset = dataset.filter(invalid_string, batched=True)
        self.assertEqual(dataset["train"].num_rows, len(data))
