import os
from pathlib import PurePosixPath
from datetime import datetime
import json

import torch
from torch import nn
from datasets import load_dataset, interleave_datasets
from transformers import BertTokenizer, BertGenerationEncoder

# checkpoint = "jvanz/querido_diario_autoencoder"
checkpoint = "neuralmind/bert-base-portuguese-cased"
MAX_SEQUENCE_LENGTH = 60
checkpoint_output_dir = "checkpoints/latent_space_classifier"

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-3


class Classifier(nn.Module):
    """
    Based on the code from @wang_controllable_2019
    """

    def __init__(self, latent_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 100)
        self.relu1 = nn.LeakyReLU(
            0.2,
        )
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out  # batch_size * label_size


def add_labels(examples):
    examples["is_gazette"] = [1.0] * len(examples["text"])
    return examples


def add_labels_is_not_gazette(examples):
    examples["is_gazette"] = [0.0] * len(examples["text"])
    return examples


def prepare_dataset(num_proc=10):
    dataset = load_dataset("jvanz/querido_diario", streaming=False)
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        ),
        num_proc=num_proc,
        batched=True,
    )
    dataset = dataset.map(add_labels, num_proc=num_proc, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "is_gazette"],
    )

    wikipedia = load_dataset("jvanz/portuguese_wikipedia_sentences")
    wikipedia = wikipedia.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        ),
        num_proc=num_proc,
        batched=True,
    )
    wikipedia = wikipedia.map(
        add_labels_is_not_gazette, num_proc=num_proc, batched=True
    )
    wikipedia.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "is_gazette"],
    )

    train_dataset = interleave_datasets(
        [
            wikipedia["train"],
            dataset["train"].select(range(wikipedia["train"].num_rows)),
        ]
    )
    test_dataset = interleave_datasets(
        [wikipedia["test"], dataset["test"].select(range(wikipedia["test"].num_rows))]
    )
    evaluation_dataset = interleave_datasets(
        [
            wikipedia["evaluation"],
            dataset["evaluation"].select(range(wikipedia["evaluation"].num_rows)),
        ]
    )

    dataset["train"] = train_dataset
    dataset["test"] = test_dataset
    dataset["evaluation"] = evaluation_dataset
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "is_gazette"],
    )

    return dataset


def train(
    encoder,
    classifier,
    dataset,
    loss_fn,
    optimizer,
    epochs: int = 100,
    batch_size: int = 64,
):
    encoder.to(device)
    classifier.to(device)
    classifier.train()

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=batch_size
    )

    min_test_error = None
    training_log = {"start_train": datetime.now().timestamp(), "epochs": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        print("-" * 50)
        losses = train_step(encoder, classifier, train_dataloader, loss_fn, optimizer)
        test_error = test_step(encoder, classifier, test_dataloader, loss_fn)

        if min_test_error is None or test_error < min_test_error:
            min_test_error = test_error
            torch.save(
                classifier,
                f"{checkpoint_output_dir}/latent_space_classifier_timestamp_{training_log['start_train']}_epoch_{epoch}_loss_{min_test_error}.pth",
            )

        epoch_log = {"epoch": epoch, "test_loss": test_error.item(), "losses": losses}
        training_log["epochs"].append(epoch_log)

    training_log["end_train"] = datetime.now().timestamp()
    return training_log


def train_step(encoder, classifier, data, loss_fn, optimizer):
    size = len(data.dataset)
    losses = []
    for batch, X in enumerate(data):
        input_ids = X["input_ids"].to(device)
        attention_mask = X["attention_mask"].to(device)
        with torch.no_grad():
            latent_space = encoder.forward(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
        latent_space = torch.sum(latent_space, dim=1)
        prediction = classifier.forward(latent_space.clone())
        labels = X["is_gazette"].to(device)
        loss = loss_fn(prediction.flatten(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        losses.append({"batch": batch, "loss": loss})
        if batch % 100 == 0:
            current = (batch + 1) * len(input_ids)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses


def test_step(encoder, classifier, data, loss_fn):
    size = len(data.dataset)
    num_batches = len(data)
    test_loss = 0.0
    with torch.no_grad():
        for X in data:
            input_ids = X["input_ids"].to(device)
            attention_mask = X["attention_mask"].to(device)
            with torch.no_grad():
                latent_space = encoder.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
            latent_space = torch.sum(latent_space, dim=1)
            prediction = classifier.forward(latent_space.clone())
            labels = X["is_gazette"].to(device)
            test_loss += loss_fn(prediction.flatten(), labels)
    test_loss /= num_batches
    print(f"Test error:\n  Avg loss: {test_loss:>8f}\n")
    return test_loss


def calculate_avg_loss(encoder, classifier, data, loss_fn):
    size = len(data.dataset)
    num_batches = len(data)
    loss = 0.0
    with torch.no_grad():
        for X in data:
            input_ids = X["input_ids"].to(device)
            attention_mask = X["attention_mask"].to(device)
            with torch.no_grad():
                latent_space = encoder.forward(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
            latent_space = torch.sum(latent_space, dim=1)
            prediction = classifier.forward(latent_space.clone())
            labels = X["is_gazette"].to(device)
            loss += loss_fn(prediction.flatten(), labels)
    loss /= num_batches
    return loss


def evaluation(encoder, classifier, dataset, loss_fn, batch_size: int = 64):
    classifier.eval()
    evaluation_dataloader = torch.utils.data.DataLoader(
        dataset["evaluation"], batch_size=batch_size
    )
    eval_loss = calculate_avg_loss(encoder, classifier, evaluation_dataloader, loss_fn)
    print(f"Evaluation error:\n  Avg loss: {eval_loss:>8f}\n")
    return eval_loss


def extract_loss_value_from_checkpoint_filepath(checkpoint):
    path = PurePosixPath(checkpoint)
    loss = path.name[: len(path.suffix) * -1].split("_")[-1]
    return float(loss)


def load_best_checkpoint():
    checkpoints = os.listdir(checkpoint_output_dir)
    min_loss = None
    min_loss_checkpoint = None
    for checkpoint in checkpoints:
        if not checkpoint.endswith("pth"):
            continue
        loss = extract_loss_value_from_checkpoint_filepath(checkpoint)
        if min_loss is None or loss < min_loss:
            min_loss = loss
            min_loss_checkpoint = checkpoint
    min_loss_checkpoint = f"{checkpoint_output_dir}/{min_loss_checkpoint}"
    print(f"Best checkpoint: {min_loss_checkpoint}")
    return torch.load(min_loss_checkpoint)


if __name__ == "__main__":
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    print(f"CUDA available? {torch.cuda.is_available()}")
    print(f"Device in use: {device}")
    dataset = prepare_dataset()

    dataset["train"] = dataset["train"].select(range(64 * 2))
    dataset["test"] = dataset["test"].select(range(64 * 2))
    dataset["evaluation"] = dataset["evaluation"].select(range(64 * 2))

    encoder = BertGenerationEncoder.from_pretrained(checkpoint)
    classifier = Classifier(encoder.config.hidden_size, 1)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

    train_json = train(encoder, classifier, dataset, loss_fn, optimizer, epochs=2)

    classifier = load_best_checkpoint()
    eval_loss = evaluation(encoder, classifier, dataset, loss_fn)
    train_json["eval_loss"] = eval_loss.item()
    with open(
        f"{checkpoint_output_dir}/training_{train_json['start_train']}.json", "w"
    ) as training_json:
        json.dump(train_json, training_json)
