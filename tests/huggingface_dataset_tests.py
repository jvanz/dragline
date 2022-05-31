import unittest
import tempfile
import string
import json
import csv
import os

import numpy as np

try:
    import datasets
    import transformers
    from datasets import load_dataset
    from transformers import (
        BertTokenizer,
        BertConfig,
        BertGenerationEncoder,
        BertGenerationDecoder,
        BertGenerationConfig,
        EncoderDecoderModel,
    )
except Exception:
    print("cannot import datasets")


def missing_huggingface():
    try:
        datasets.__version__
        transformers.__version__
    except:
        return True
    return False


def valid_string(string_value):
    translate_table = str.maketrans("", "", string.punctuation)
    return len(string_value.translate(translate_table).strip()) > 0


@unittest.skipIf(missing_huggingface(), "No Huggingface lib detected. Skipping test.")
class HuggingFacesModelTests(unittest.TestCase):
    def test_bert_generation(self):
        model_name_or_path = "neuralmind/bert-base-portuguese-cased"
        model_configuration_values = {
            "return_dict": True,
            "architectures": ["BertForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "directionality": "bidi",
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "output_past": True,
            "pad_token_id": 0,
            "pooler_fc_size": 768,
            "pooler_num_attention_heads": 12,
            "pooler_num_fc_layers": 3,
            "pooler_size_per_head": 128,
            "pooler_type": "first_token_transform",
            "type_vocab_size": 2,
            "vocab_size": 29794,
        }

        encoder = BertGenerationEncoder.from_pretrained(model_name_or_path)
        for key in model_configuration_values:
            self.assertEqual(
                encoder.config.to_dict()[key],
                model_configuration_values[key],
                f"Unexpected {key} value",
            )
        decoder = BertGenerationDecoder.from_pretrained(model_name_or_path)
        for key in model_configuration_values:
            self.assertEqual(
                encoder.config.to_dict()[key],
                model_configuration_values[key],
                f"Unexpected {key} value",
            )

        encoder_decoder_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_name_or_path, model_name_or_path
        )
        for key in model_configuration_values:
            self.assertEqual(
                encoder_decoder_model.encoder.config.to_dict()[key],
                model_configuration_values[key],
                f"Unexpected {key} value",
            )
            self.assertEqual(
                encoder_decoder_model.decoder.config.to_dict()[key],
                model_configuration_values[key],
                f"Unexpected {key} value",
            )
        self.assertTrue(encoder_decoder_model.decoder.config.is_decoder)


@unittest.skipIf(missing_huggingface(), "No Huggingface lib detected. Skipping test.")
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
