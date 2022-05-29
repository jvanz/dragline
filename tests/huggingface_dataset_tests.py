import unittest
import tempfile
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
