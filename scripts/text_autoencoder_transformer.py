from dataclasses import dataclass, field
import argparse
import csv
import json
import logging
import os
import pathlib
import string

from transformers import (
    EncoderDecoderModel,
    BertGenerationEncoder,
    BertGenerationDecoder,
    AutoTokenizer,
    BertTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
)
from datasets import load_metric, load_dataset
from datasets.fingerprint import Hasher

import numpy as np
import torch
from torch.utils.data import DataLoader

from gazettes.transformer import BertDataset

EPOCHS = int(os.environ.get("EPOCHS", 10))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: str = field(metadata={"help": "Pretrained tokenizer name or path"})
    resume_training: bool = field(default=True)


@dataclass
class DataArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    data_dir: str = field(
        metadata={"help": "Path to directory storing the dataset files"}
    )
    dataset_partial_load: float = field(default=1.0)
    shuffle_dataset: bool = field(default=True)


def create_model(tokenizer, model_name_or_path):
    encoder = BertGenerationEncoder.from_pretrained(
        model_name_or_path, bos_token_id=103, eos_token_id=102
    )

    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    decoder = BertGenerationDecoder.from_pretrained(
        model_name_or_path,
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


def tokenize_function(examples, tokenizer=None):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=20,
        add_special_tokens=True,
    )


def valid_string(string_value):
    translate_table = str.maketrans("", "", string.punctuation)
    return len(string_value.translate(translate_table).strip()) > 0


def filter_invalid_string(examples):
    return [valid_string(example) for example in examples["text"]]


def reorganize_features(examples):
    examples["decoder_input_ids"] = examples["input_ids"]
    examples["labels"] = examples["input_ids"]
    return examples


def load_huggingface_datasets(data_dir, tokenizer, partial_dataset_load, shuffle):

    train_dataset = load_dataset("csv", data_files=f"{data_dir}/train.csv")
    train_dataset = train_dataset.filter(
        filter_invalid_string, batched=True, num_proc=os.cpu_count()
    )
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        fn_kwargs={"tokenizer": tokenizer},
    )
    train_dataset = train_dataset.map(
        reorganize_features,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text", "token_type_ids", "attention_mask"],
    )
    if shuffle:
        train_dataset = train_dataset.shuffle()
    train_dataset = train_dataset["train"].select(
        range(int(train_dataset["train"].num_rows * partial_dataset_load))
    )

    evaluation_dataset = load_dataset("csv", data_files=f"{data_dir}/evaluation.csv")
    evaluation_dataset = evaluation_dataset.filter(
        filter_invalid_string, batched=True, num_proc=os.cpu_count()
    )
    evaluation_dataset = evaluation_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        fn_kwargs={"tokenizer": tokenizer},
    )
    evaluation_dataset = evaluation_dataset.map(
        reorganize_features,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text", "token_type_ids", "attention_mask"],
    )
    if shuffle:
        evaluation_dataset = evaluation_dataset.shuffle()
    evaluation_dataset = evaluation_dataset["train"].select(
        range(int(evaluation_dataset["train"].num_rows * partial_dataset_load))
    )

    test_dataset = load_dataset("csv", data_files=f"{data_dir}/test.csv")
    test_dataset = test_dataset.filter(
        filter_invalid_string, batched=True, num_proc=os.cpu_count()
    )
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        fn_kwargs={"tokenizer": tokenizer},
    )
    test_dataset = test_dataset.map(
        reorganize_features,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text", "token_type_ids", "attention_mask"],
    )
    if shuffle:
        test_dataset = test_dataset.shuffle()
    test_dataset = test_dataset["train"].select(
        range(int(test_dataset["train"].num_rows * partial_dataset_load))
    )

    return train_dataset, evaluation_dataset, test_dataset


def load_datasets(data_dir, tokenizer, batch_size: int = 16):
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
        DataLoader(train_dataset, batch_size=16),
        DataLoader(eval_dataset, batch_size=16),
        DataLoader(test_dataset, batch_size=16),
    )


def get_checkpoint_dir(model):
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def train_loop(dataset, model, loss_fn, optimizer, device, epoch, checkpoint_dir):
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                f"{checkpoint_dir}/model.pt",
            )


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


def train_torch_model(
    model, train_dataset, eval_dataset, test_dataset, device, checkpoint_dir: str = None
):
    loss_fn = torch.nn.MSELoss
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}\n--------------------------")
        train_loop(
            train_dataset, model, loss_fn, optimizer, device, epoch, checkpoint_dir
        )
        test_loop(eval_dataset, model, loss_fn)
        torch.save(model, "models/text_transformer_autoencoder.pth")
    print("Done!")


def parse_command_line_arguments():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    return parser.parse_args_into_dataclasses()


def main():
    logging.basicConfig(level=logging.DEBUG)
    model_args, data_args, training_args = parse_command_line_arguments()
    logging.debug(model_args)
    logging.debug(data_args)
    logging.debug(training_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    os.makedirs(training_args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        logging.info("CUDA is available")
        current_device = torch.cuda.get_device_name(torch.cuda.current_device())
        logging.info(f"Device in use: {current_device}")
        logging.info(f"CUDA version: {torch.version.cuda}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(
        model_args.model_name_or_path,
        do_lower_case=False,
        use_fast=False,
        bos_token_id=101,
        eos_token_id=102,
    )

    train_dataset, eval_dataset, test_dataset = load_huggingface_datasets(
        data_args.data_dir,
        tokenizer,
        data_args.dataset_partial_load,
        data_args.shuffle_dataset,
    )
    logging.debug(train_dataset)
    logging.debug(train_dataset[0])
    logging.debug(eval_dataset)
    logging.debug(eval_dataset[0])
    logging.debug(test_dataset)
    logging.debug(test_dataset[0])

    model = create_model(tokenizer, model_args.model_name_or_path)
    model.to(device)
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=model_args.resume_training)
    # train_torch_model(
    #     model,
    #     train_dataset,
    #     eval_dataset,
    #     test_dataset,
    #     device,
    #     checkpoint_dir=args.checkpoint_dir,
    # )


if __name__ == "__main__":
    main()
