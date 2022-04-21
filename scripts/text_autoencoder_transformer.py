import os
import logging
import json

from transformers import (
    TFBertModel,
    TFBertForMaskedLM,
    TFBertLMHeadModel,
    TFBertForSequenceClassification,
    TFEncoderDecoderModel,
    EncoderDecoderModel,
    BertGenerationEncoder,
    BertGenerationDecoder,
    AutoTokenizer,
    BertTokenizer,
)

# import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import DataLoader

# from gensim.models import KeyedVectors

from gazettes.data import WikipediaDataset

WIKIPEDIA_DATA_DIR = str(os.environ.get("WIKIPEDIA_DATA_DIR", "data/wikipedia"))
WIKIPEDIA_DATASET_SIZE = float(os.environ.get("WIKIPEDIA_DATASET_SIZE", 1.0))
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 64))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS", tf.data.AUTOTUNE))
DIMENSOES_ESPACO_LATENTE = int(os.environ.get("DIMENSOES_ESPACO_LATENTE", 32))
DEFAULT_MODEL_NAME = "text_transformer_autoencoder"
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT", "neuralmind/bert-base-portuguese-cased"
)
MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
# NUM_PARALLEL_CALLS = int(os.environ.get("NUM_PARALLEL_CALLS", tf.data.AUTOTUNE))
VOCAB_FILE = os.environ.get("VOCAB_FILE", "data/bertimbau_base_vocab.txt")
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 4096))
WIKIPEDIA_DATASET_SIZE = float(os.environ.get("WIKIPEDIA_DATASET_SIZE", 1.0))
WIKIPEDIA_DATA_DIR = str(os.environ.get("WIKIPEDIA_DATA_DIR", "data/wikipedia"))

PATIENCE = 20


class BertDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir: str, dataset_dir: str, tokenizer):
        self.data_dir = f"{data_dir}/{dataset_dir}"
        self.tokenizer = tokenizer
        with open(f"{data_dir}/metadata.json", "r") as jsonfile:
            metadata = json.load(jsonfile)
            self.length = metadata[dataset_dir].get("length", 0)

    def __len__(self):
        return self.length

    def __iter__(self):
        dataset = WikipediaDataset(self.data_dir)
        for batch in dataset.as_numpy_iterator():
            for sample in batch:
                input_ids = self.tokenizer(
                    sample.decode("utf8"),
                    add_special_tokens=True,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=40,
                ).input_ids[0]
                yield input_ids, input_ids


def create_model(tokenizer):
    encoder = BertGenerationEncoder.from_pretrained(
        MODEL_CHECKPOINT, bos_token_id=103, eos_token_id=102
    )

    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    decoder = BertGenerationDecoder.from_pretrained(
        MODEL_CHECKPOINT,
        add_cross_attention=True,
        is_decoder=True,
        bos_token_id=101,
        eos_token_id=102,
    )
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    return model


def load_datasets():
    tokenizer = BertTokenizer.from_pretrained(
        MODEL_CHECKPOINT,
        do_lower_case=False,
        use_fast=False,
        bos_token_id=101,
        eos_token_id=102,
    )
    train_dataset = BertDataset("data/wikipedia", "train", tokenizer)
    print(f"len(train_dataset): {len(train_dataset)}")
    eval_dataset = BertDataset("data/wikipedia", "evaluation", tokenizer)
    print(f"len(eval_dataset): {len(eval_dataset)}")
    test_dataset = BertDataset("data/wikipedia", "test", tokenizer)
    print(f"len(test_dataset): {len(test_dataset)}")
    return (
        DataLoader(train_dataset, batch_size=32),
        DataLoader(eval_dataset, batch_size=32),
        DataLoader(test_dataset, batch_size=32),
        tokenizer,
    )


def get_checkpoint_dir(model):
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def train_loop(dataset, model, loss_fn, optimizer):
    size = len(dataset.dataset)
    for batch, (inputs, expected_prediction) in enumerate(dataset):
        output = model(input_ids=inputs, labels=inputs)
        loss = output.loss

        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataset, model, loss_fn):
    size = len(dataset.dataset)
    num_batches = len(dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, expected_prediction in dataset:
            predictions = model(inputs)
            test_loop += loss_fn(predictions, expected_prediction).item()
            correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
    test_loop /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def train_torch_model(model, train_dataset, eval_dataset, test_dataset):
    loss_fn = torch.nn.MSELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}\n--------------------------")
        train_loop(train_dataset, model, loss_fn, optimizer)
        test_loop(eval_dataset, model, loss_fn)
        torch.save(model, "models/text_transformer_autoencoder.pth")
    print("Done!")


def main():
    logging.basicConfig(level=logging.INFO)

    train_dataset, eval_dataset, test_dataset, tokenizer = load_datasets()
    model = create_model(tokenizer)
    train_torch_model(model, train_dataset, eval_dataset, test_dataset)


if __name__ == "__main__":
    main()
