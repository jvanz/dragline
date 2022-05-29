import unittest
import tempfile
import json
import csv
import os

try:
    import torch
except Exception:
    print("cannot import pytorch")
import numpy as np
from transformers import BertTokenizer

from gazettes.transformer import BertDataset


def missing_pytorch():
    try:
        torch.__version__
    except:
        return True
    return False


@unittest.skipIf(missing_pytorch(), "No Pytorch detected. Skipping test.")
class TorchBertDataset(unittest.TestCase):

    model_checkpoint = "neuralmind/bert-base-portuguese-cased"

    def write_csv_file(self, data):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=False
        ) as csv_file:
            fieldnames = list(data.keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for column in data:
                for row in data[column]:
                    writer.writerow({column: row})
            csv_file.flush()
            return csv_file.name

    def write_metadata_file(self, metadata):
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf8", delete=False
        ) as metadata_file:
            json.dump(metadata, metadata_file)
            metadata_file.flush()
            return metadata_file.name

    def write_csv_and_metadata(self, data):
        csv_file = self.write_csv_file({"text": data})
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
        metadatafile = self.write_metadata_file({dataset_name: {"length": len(data)}})
        return csv_file, metadatafile

    def test_load_csv_file_into_dataset(self):
        data = ["This is the first row", "This is the second row"]
        csv_file, metadatafile = self.write_csv_and_metadata(data)
        dataset = BertDataset(csv_file, metadatafile=metadatafile)
        data = list(dataset)
        self.assertEqual(data, ["This is the first row", "This is the second row"])

    def test_add_target(self):
        data = ["This is the first row", "This is the second row"]
        csv_file, metadatafile = self.write_csv_and_metadata(data)
        dataset = BertDataset(csv_file, metadatafile=metadatafile, add_target=True)
        data = list(dataset)
        self.assertEqual(
            data,
            [
                ("This is the first row", "This is the first row"),
                ("This is the second row", "This is the second row"),
            ],
        )

    def test_tokenize_dataset(self):
        data = ["This is the first row"]
        csv_file, metadatafile = self.write_csv_and_metadata(data)
        tokenizer = BertTokenizer.from_pretrained(
            self.model_checkpoint,
            do_lower_case=False,
            use_fast=False,
            bos_token_id=101,
            eos_token_id=102,
        )
        data = list(
            BertDataset(
                csv_file,
                tokenizer=tokenizer,
                metadatafile=metadatafile,
            )
        )
        expected_data = [
            torch.tensor([101, 16989, 847, 1621, 5101, 7485, 577, 22343, 102])
        ]
        for output, expected in zip(data, expected_data):
            self.assertTrue(torch.equal(output, expected))

    def test_padding_truncation_dataset(self):
        data = [
            "This is the first row",
            "This is the first row This is the first row This is the first row",
        ]
        csv_file, metadatafile = self.write_csv_and_metadata(data)
        tokenizer = BertTokenizer.from_pretrained(
            self.model_checkpoint,
            do_lower_case=False,
            use_fast=False,
            bos_token_id=101,
            eos_token_id=102,
        )
        data = list(
            BertDataset(
                csv_file,
                tokenizer=tokenizer,
                max_sequence_length=11,
                truncation=True,
                padding="max_length",
                metadatafile=metadatafile,
            )
        )
        expected_data = [
            torch.tensor([101, 16989, 847, 1621, 5101, 7485, 577, 22343, 102, 0, 0]),
            torch.tensor(
                [101, 16989, 847, 1621, 5101, 7485, 577, 22343, 16989, 847, 102]
            ),
        ]
        for output, expected in zip(data, expected_data):
            self.assertTrue(torch.equal(output, expected))

    def test_load_metadata_file(self):
        data = ["This is the first row", "This is the second row"]
        csv_file, metadatafile = self.write_csv_and_metadata(data)
        dataset = BertDataset(csv_file, metadatafile=metadatafile)
        data = list(dataset)
        self.assertEqual(data, ["This is the first row", "This is the second row"])
        self.assertEqual(len(dataset), 2)
