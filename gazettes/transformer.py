import os
import csv
import json

import torch


class BertDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        datafile: str,
        metadatafile: str = None,
        tokenizer=None,
        max_sequence_length: int = None,
        truncation: bool = None,
        padding: str = None,
        add_target: bool = False,
    ):
        assert os.path.exists(datafile)
        self.datafile = datafile
        self.tokenizer = tokenizer
        self.add_target = add_target
        self.tokenizer_args = {"add_special_tokens": True, "return_tensors": "pt"}
        if max_sequence_length:
            self.tokenizer_args["max_length"] = max_sequence_length
        if truncation:
            self.tokenizer_args["truncation"] = truncation
        if padding:
            self.tokenizer_args["padding"] = padding
        if metadatafile:
            with open(metadatafile, "r") as metadata:
                self.metadata = json.load(metadata)

    def __len__(self):
        assert self.metadata is not None
        dataset_name = os.path.splitext(os.path.basename(self.datafile))[0]
        return self.metadata[dataset_name]["length"]

    def __iter__(self):
        with open(self.datafile, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sample = row["text"]
                if self.tokenizer:
                    sample = self.tokenizer(sample, **self.tokenizer_args).input_ids[0]
                if self.add_target:
                    yield sample, sample
                else:
                    yield sample
