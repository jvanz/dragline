import os

import torch
from torch import nn
from datasets import load_dataset, interleave_datasets
from transformers import BertTokenizer, BertGenerationEncoder

# checkpoint = "jvanz/querido_diario_autoencoder"
checkpoint = "neuralmind/bert-base-portuguese-cased"
MAX_SEQUENCE_LENGTH = 60
checkpoint_output_dir = "checkpoints/latent_space_classifier"

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    print(dataset)
    print(dataset["train"][:3])

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
    print(wikipedia)
    print(wikipedia["train"][:3])

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
    print(dataset)
    print(dataset["train"][:3])
    print("-" * 100)

    return dataset


def train_step(encoder, classifier, data, loss_fn, optimizer):
    size = len(data.dataset)
    for batch, X in enumerate(data):
        input_ids = X["input_ids"].to(device)
        attention_mask = X["attention_mask"].to(device)
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def eval_step(encoder, classifier, data, loss_fn):
    size = len(data.dataset)
    num_batches = len(data)
    test_loss = 0.0
    with torch.no_grad():
        for X in data:
            input_ids = X["input_ids"].to(device)
            attention_mask = X["attention_mask"].to(device)
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


def train(
    dataset, epochs: int = 100, batch_size: int = 64, learning_rate: float = 1e-3
):
    encoder = BertGenerationEncoder.from_pretrained(checkpoint)
    encoder.to(device)
    classifier = Classifier(encoder.config.hidden_size, 1)
    classifier.to(device)
    classifier.train()

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=batch_size
    )

    min_test_error = None
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        print("-" * 50)
        train_step(encoder, classifier, train_dataloader, loss_fn, optimizer)
        test_error = eval_step(encoder, classifier, test_dataloader, loss_fn)

        if min_test_error is None or test_error < min_test_error:
            min_test_error = test_error
            torch.save(
                classifier,
                f"{checkpoint_output_dir}/latent_space_classifier_{min_test_error}.pth",
            )
        return


if __name__ == "__main__":
    os.makedirs(checkpoint_output_dir, exist_ok=True)
    print(f"CUDA available? {torch.cuda.is_available()}")
    print(f"Device in use: {device}")
    dataset = prepare_dataset()
    train(dataset)
