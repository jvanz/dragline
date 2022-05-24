import os
import logging
import argparse
import json
import pathlib
import csv

from transformers import (
    EncoderDecoderModel,
    BertGenerationEncoder,
    BertGenerationDecoder,
    AutoTokenizer,
    BertTokenizer,
)

import numpy as np
import torch
from torch.utils.data import DataLoader

from gazettes.transformer import BertDataset

EPOCHS = int(os.environ.get("EPOCHS", 10))
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT", "neuralmind/bert-base-portuguese-cased"
)


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


def load_datasets(data_dir, tokenizer):
    train_dataset = BertDataset(
        f"{data_dir}/train.csv",
        metadatafile=f"{data_dir}/metadata.json",
        tokenizer=tokenizer,
        max_sequence_length=20,
        truncation=True,
        padding="max_length",
        add_target=True,
    )
    eval_dataset = BertDataset(
        f"{data_dir}/evaluation.csv",
        metadatafile=f"{data_dir}/metadata.json",
        tokenizer=tokenizer,
        max_sequence_length=20,
        truncation=True,
        padding="max_length",
        add_target=True,
    )
    test_dataset = BertDataset(
        f"{data_dir}/test.csv",
        metadatafile=f"{data_dir}/metadata.json",
        tokenizer=tokenizer,
        max_sequence_length=20,
        truncation=True,
        padding="max_length",
        add_target=True,
    )
    return (
        DataLoader(train_dataset, batch_size=32),
        DataLoader(eval_dataset, batch_size=32),
        DataLoader(test_dataset, batch_size=32),
    )


def get_checkpoint_dir(model):
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def train_loop(dataset, model, loss_fn, optimizer, device):
    size = len(dataset.dataset)
    for batch, (inputs, expected_prediction) in enumerate(dataset):
        inputs = inputs.to(device)
        expected_prediction = expected_prediction.to(device)

        output = model(input_ids=inputs, decoder_input_ids=inputs, labels=inputs)
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


def train_torch_model(model, train_dataset, eval_dataset, test_dataset, device):
    loss_fn = torch.nn.MSELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}\n--------------------------")
        train_loop(train_dataset, model, loss_fn, optimizer, device)
        test_loop(eval_dataset, model, loss_fn)
        torch.save(model, "models/text_transformer_autoencoder.pth")
    print("Done!")


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        type=pathlib.Path,
    )
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_command_line_arguments()

    if torch.cuda.is_available():
        logging.info("CUDA is available")
        current_device = torch.cuda.get_device_name(torch.cuda.current_device())
        logging.info(f"Device in use: {current_device}")
        logging.info(f"CUDA version: {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(
        MODEL_CHECKPOINT,
        do_lower_case=False,
        use_fast=False,
        bos_token_id=101,
        eos_token_id=102,
    )

    train_dataset, eval_dataset, test_dataset = load_datasets(args.data_dir, tokenizer)
    model = create_model(tokenizer)
    model.to(device)
    train_torch_model(model, train_dataset, eval_dataset, test_dataset, device)


if __name__ == "__main__":
    main()
