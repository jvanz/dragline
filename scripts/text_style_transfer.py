import pathlib
import os
import argparse
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, EncoderDecoderModel
from datasets import load_dataset, interleave_datasets

logger = logging.getLogger()

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


def commandline_arguments_parsing():
    parser = argparse.ArgumentParser(
        description="Script to train text style transfer models"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs count")
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints"),
        help="",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name. User to save the model checkpoints in the checkpoint dir",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=60,
        help="",
    )
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available(), help="")
    parser.add_argument(
        "--device", type=str, default=device, help="Device used to train"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument(
        "--classifier-labels",
        type=int,
        default=1,
        help="How many label the classifier should be able to detect",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size to be used in training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate to be used in training steps",
    )

    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.joinpath(args.name)
    return args


def print_start_epoch_header(epoch: int):
    assert epoch >= 0
    print(f"\nEpoch {epoch}")
    print("_" * 50)


def save_models(
    autoencoder: EncoderDecoderModel,
    autoencoder_loss: float,
    classifier: nn.Module,
    classifier_loss: float,
    checkpoint_dir: str,
    epoch: int,
):
    torch.save(
        autoencoder,
        f"{checkpoint_dir}/autoencoder_{epoch}_{autoencoder_loss}.pth",
    )
    torch.save(
        classifier,
        f"{checkpoint_dir}/classifier_{epoch}_{classifier_loss}.pth",
    )


def train(
    epochs,
    autoencoder,
    classifier,
    datasets,
    batch_size,
    learning_rate,
    device,
    checkpoint_dir,
    tokenizer,
    **kw_args,
):
    print("-" * 100)
    print(f"Starting training...")
    print(f"Device used: {device}")
    train_dataloader = DataLoader(datasets["train"], batch_size=batch_size)
    test_dataloader = DataLoader(datasets["test"], batch_size=batch_size)
    evaluation_dataloader = DataLoader(datasets["evaluation"], batch_size=batch_size)

    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    autoencoder.zero_grad()
    autoencoder.to(device)

    classifier.to(device)
    classifier.zero_grad()
    classifier_loss_fn = nn.BCELoss()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print_start_epoch_header(epoch)
        train_step(
            autoencoder,
            autoencoder_optimizer,
            classifier,
            classifier_loss_fn,
            classifier_optimizer,
            train_dataloader,
            device,
        )
        autoencoder_loss, classifier_loss = evaluation_step(
            autoencoder,
            classifier,
            classifier_loss_fn,
            test_dataloader,
            device,
            tokenizer,
        )
        save_models(
            autoencoder,
            autoencoder_loss,
            classifier,
            classifier_loss,
            checkpoint_dir,
            epoch,
        )
    print("Training finished.")
    evaluation_step(
        autoencoder,
        classifier,
        classifier_loss_fn,
        evaluation_dataloader,
        device,
        tokenizer,
        prefix_text="Evaluation losses",
    )


def train_step(
    autoencoder,
    autoencoder_optimizer,
    classifier,
    classifier_loss_fn,
    classifier_optimizer,
    dataset,
    device,
):
    autoencoder.train()
    classifier.train()
    current = 1
    size = dataset.dataset.num_rows
    batch_size = dataset.batch_size
    for batch in dataset:
        # autoencoder_optimizer.zero_grad()
        # classifier_optimizer.zero_grad()
        autoencoder.zero_grad()
        classifier.zero_grad()

        # train autoencoder
        autoencoder_output = autoencoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["input_ids"].to(device),
        )
        autoencoder_output.loss.backward()
        autoencoder_optimizer.step()

        encoder_latent_space = torch.sum(
            autoencoder_output.encoder_last_hidden_state.detach(), dim=1
        )

        logger.debug(
            f"Encoder last hidden state size: {autoencoder_output.encoder_last_hidden_state.size()}"
        )
        logger.debug(f"Encoder last hidden state size: {encoder_latent_space.size()}")
        # train classifier
        classifier_output = classifier.forward(encoder_latent_space)
        classifier_loss = classifier_loss_fn(
            classifier_output.flatten(), batch["is_legal"].to(device)
        )
        classifier_loss.backward()
        classifier_optimizer.step()

        if current % 100 == 0:
            steps = current * batch_size
            print(
                f"Autoencoder loss: {autoencoder_output.loss.item():>8f}, classifier loss: {classifier_loss.item():>8f}  [{steps}/{size}] "
            )

        current += 1


def evaluation_step(
    autoencoder,
    classifier,
    classifier_loss_fn,
    dataset,
    device,
    tokenizer,
    prefix_text="Test losses",
):
    size = int(dataset.dataset.num_rows / dataset.batch_size)
    autoencoder_loss = 0
    classifier_loss = 0
    current = 0
    with torch.no_grad():
        for batch in dataset:
            current += 1

            # test autoencoder
            autoencoder_output = autoencoder.forward(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
            autoencoder_loss += autoencoder_output.loss.item()
            encoder_latent_space = torch.sum(
                autoencoder_output.encoder_last_hidden_state.detach(), dim=1
            )

            input_sentence = batch["input_ids"][0].to(device)
            output_sentence = autoencoder.generate(input_sentence.unsqueeze(0))[0].to(
                device
            )
            input_sentence = tokenizer.decode(
                batch["input_ids"][0], skip_special_tokens=True
            )
            output_sentence = tokenizer.decode(
                output_sentence, skip_special_tokens=True
            )

            print(f"    Autoencoder input [{current}/{size}]: {input_sentence}")
            print(f"    Autoencoder output [{current}/{size}]: {output_sentence}")

            # test classifier
            classifier_output = classifier.forward(encoder_latent_space)
            classifier_loss_value = classifier_loss_fn(
                classifier_output.flatten(), batch["is_legal"].to(device)
            )
            classifier_loss += classifier_loss_value.item()

    autoencoder_loss /= dataset.batch_size
    classifier_loss /= dataset.batch_size
    print(
        f"{prefix_text}:\n    Avg autoencoder loss: {autoencoder_loss:>8f}, Avg classifier loss: {classifier_loss:>8f}"
    )
    return autoencoder_loss, classifier_loss


def prepare_dataset(tokenizer, max_sequence_length, num_proc=10):
    assert tokenizer != None
    querido_diario = load_dataset("jvanz/querido_diario", streaming=False)
    querido_diario = querido_diario.map(
        lambda x: {"is_legal": [1.0] * len(x["text"])}, num_proc=num_proc, batched=True
    )
    wikipedia = load_dataset("jvanz/portuguese_wikipedia_sentences")
    wikipedia = wikipedia.map(
        lambda x: {"is_legal": [0.0] * len(x["text"])}, num_proc=num_proc, batched=True
    )

    train_dataset = interleave_datasets(
        [
            wikipedia["train"],
            querido_diario["train"].select(range(wikipedia["train"].num_rows)),
        ]
    )
    test_dataset = interleave_datasets(
        [
            wikipedia["test"],
            querido_diario["test"].select(range(wikipedia["test"].num_rows)),
        ]
    )
    evaluation_dataset = interleave_datasets(
        [
            wikipedia["evaluation"],
            querido_diario["evaluation"].select(
                range(wikipedia["evaluation"].num_rows)
            ),
        ]
    )

    querido_diario["train"] = train_dataset
    querido_diario["test"] = test_dataset
    querido_diario["evaluation"] = evaluation_dataset
    querido_diario = querido_diario.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        ),
        num_proc=num_proc,
        batched=True,
    )
    querido_diario.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "is_legal"],
    )
    querido_diario = querido_diario.shuffle(seed=42)
    logger.info(querido_diario)
    return querido_diario


def load_tokenizer(checkpoint):
    return BertTokenizer.from_pretrained(checkpoint)


def create_models(checkpoint: str, classifier_labels: int, tokenizer: BertTokenizer):
    assert checkpoint != None and len(checkpoint) > 0
    assert classifier_labels > 0
    autoencoder = EncoderDecoderModel.from_encoder_decoder_pretrained(
        checkpoint, checkpoint
    )
    autoencoder.config.decoder_start_token_id = tokenizer.cls_token_id
    autoencoder.config.pad_token_id = tokenizer.cls_token_id
    classifier = Classifier(autoencoder.config.encoder.hidden_size, classifier_labels)
    logger.debug(autoencoder)
    logger.debug(classifier)
    return autoencoder, classifier


if __name__ == "__main__":
    config = commandline_arguments_parsing()
    logging.basicConfig(level=logging.DEBUG if config.debug else logging.INFO)
    logger.debug(config)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer = load_tokenizer(config.checkpoint)
    datasets = prepare_dataset(tokenizer, config.max_sequence_length)
    autoencoder, classifier = create_models(
        config.checkpoint, config.classifier_labels, tokenizer
    )
    if config.train:
        train(
            **vars(config),
            autoencoder=autoencoder,
            classifier=classifier,
            datasets=datasets,
            tokenizer=tokenizer,
        )
