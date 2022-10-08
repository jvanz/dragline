import pathlib
import os
import argparse
import logging
import shutil
from datetime import datetime
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    BertGenerationEncoder,
    BertGenerationDecoder,
)
from datasets import load_dataset, concatenate_datasets
import numpy as np

logger = logging.getLogger()

device = "cuda" if torch.cuda.is_available() else "cpu"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class FeedforwardClassifier(nn.Module):
    def __init__(self, latent_size: int, output_size: int, num_layers=8):
        super(FeedforwardClassifier, self).__init__()

        layers_config = np.linspace(latent_size, output_size, num=num_layers, dtype=int)
        self.model = nn.Sequential(
            nn.BatchNorm1d(latent_size), nn.Linear(layers_config[0], layers_config[1])
        )
        for index, config in enumerate(layers_config[1:], start=1):
            if index + 1 == len(layers_config):
                break
            self.model.append(nn.Dropout(p=0.2))
            self.model.append(nn.Linear(config, layers_config[index + 1]))
            if index < len(layers_config) - 2:
                self.model.append(nn.GELU())

    def forward(self, input):
        out = self.model(input)
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
        "--checkpoint",
        type=str,
        required=False,
        default="jvanz/querido_diario_autoencoder",
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
        default=5e-3,
        help="Learning rate to be used in training steps",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of steps to see some loss decrease",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="The minimum loss value should be decrease after the steps defined at --patience",
    )
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=1000,
        help="Define how many step should be wait until perform evaluation",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Define how many step should be wait until log training info (e.g. losses values)",
    )
    parser.add_argument(
        "--dataset-name",
        choices=["legal", "sentiment"],
        required=True,
        type=str,
        help="Define the dataset to be used in the training",
    )

    args = parser.parse_args()
    return args


def save_models(
    classifier: nn.Module,
    classifier_loss: float,
    checkpoint_dir: str,
    step: int,
    model_name: str,
):
    torch.save(
        classifier,
        f"{checkpoint_dir}/{model_name}_{step}_{classifier_loss}.pth",
    )


def load_datasets(datasets, batch_size):
    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )
    evaluation_dataloader = DataLoader(
        datasets["validation"],
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )
    return train_dataloader, evaluation_dataloader


def train_one_epoch(
    autoencoder, classifier, train_dataloader, loss_fn, optimizer, epoch, summary=None
):
    # import pdb

    # pdb.set_trace()
    autoencoder = autoencoder.train(False)
    classifier = classifier.train(True)
    running_loss = 0.0
    epoch_optimization_steps = int(
        train_dataloader.dataset.num_rows / train_dataloader.batch_size
    )
    for i, batch in enumerate(train_dataloader):
        global_step = (i + 1) + (epoch * epoch_optimization_steps)
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        autoencoder_output = autoencoder(
            input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids
        )

        latent_space = torch.sum(autoencoder_output.encoder_last_hidden_state, dim=1)
        classifier_output = classifier(latent_space)

        loss = loss_fn(
            classifier_output.flatten(),
            batch["label"].to(device),
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        summary.add_scalar("Training loss", loss.item(), global_step)
        if epoch == 0:
            summary.add_graph(classifier, latent_space)
        print(
            f"Loss: {loss.item():>8f} [{i+1}/{epoch_optimization_steps}] Epoch {epoch+1} [{global_step}]"
        )

    return running_loss / epoch_optimization_steps


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
    early_stopping_patience,
    early_stopping_threshold,
    evaluation_steps,
    logging_steps,
    model_name,
    **kw_args,
):
    print("-" * 100)
    print(f"Starting training...")
    print(f"Device used: {device}")
    train_dataloader, evaluation_dataloader = load_datasets(datasets, batch_size)

    autoencoder.to(device)
    classifier.to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    classifier_loss_fn = nn.BCEWithLogitsLoss().to(device)

    optimization_steps = (
        int(train_dataloader.dataset.num_rows / train_dataloader.batch_size) * epochs
    )

    writer = SummaryWriter(f"runs/{model_name}_{timestamp}")
    writer.add_text(
        "Training configuration",
        f"Optimization steps: {optimization_steps}\n\n"
        + f"Epochs: {epochs}\n\n"
        + f"Train dataset size: {train_dataloader.dataset.num_rows}\n\n"
        + f"Validation dataset size: {evaluation_dataloader.dataset.num_rows}\n\n"
        + f"Batch size: {train_dataloader.batch_size}\n\n"
        + f"Loss function: {classifier_loss_fn}\n\n"
        + f"Learning rate: {learning_rate}\n\n"
        + f"Early stopping patience: {early_stopping_patience}\n\n"
        + f"Early stopping threshold: {early_stopping_threshold}\n\n"
        + f"Optimizer: {classifier_optimizer}",
    )

    print(f"Total optimization steps: {optimization_steps}")
    print(f"Train dataset size: {train_dataloader.dataset.num_rows}")
    print(f"Batch size: {train_dataloader.batch_size}")

    best_classifier_loss = None
    patience = early_stopping_patience

    for epoch in range(epochs):
        print("#" * 3 + f" Epoch [{epoch+1}/{epochs}] " + "#" * 3)
        avg_train_loss = train_one_epoch(
            autoencoder,
            classifier,
            train_dataloader,
            classifier_loss_fn,
            classifier_optimizer,
            epoch,
            writer,
        )
        avg_validation_loss = validation(
            autoencoder, classifier, evaluation_dataloader, classifier_loss_fn
        )

        # write the values into tensorboard
        writer.add_scalars(
            "Classifier train",
            {
                "Train": avg_train_loss,
                "Validation": avg_validation_loss,
            },
            epoch + 1,
        )

        # new best loss value?
        if best_classifier_loss is None or avg_validation_loss < best_classifier_loss:
            best_classifier_loss = avg_validation_loss
            save_models(
                classifier,
                avg_validation_loss,
                checkpoint_dir,
                epoch,
                model_name,
            )
            patience = early_stopping_patience
        else:
            patience -= 1

        if patience == 0:
            print(
                f"Halter training at {epoch+1}. No loss decreasing for {early_stopping_patience} epochs. Halting the training."
            )
            writer.add_text(
                "Halted training",
                f"Halter training at {epoch+1}. No loss decreasing for {early_stopping_patience} epochs. Halting the training.",
            )
            break

    writer.close()
    print("Training finished.")


def validation(autoencoder, classifier, evaluation_dataloader, classifier_loss_fn):
    running_loss = 0.0
    classifier.eval()
    with torch.no_grad():
        for batch in evaluation_dataloader:
            autoencoder_output = autoencoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
            latent_space = torch.sum(
                autoencoder_output.encoder_last_hidden_state, dim=1
            )
            classifier_output = classifier(latent_space)

            classifier_loss = classifier_loss_fn(
                classifier_output.flatten(), batch["label"].to(device)
            )
            running_loss += classifier_loss.item()

    avg_validation_loss = running_loss / len(evaluation_dataloader)
    print(f"Avg validation loss: {avg_validation_loss}")

    return avg_validation_loss


def prepare_dataset(tokenizer, max_sequence_length, dataset_name: str, num_proc=10):
    dataset = load_dataset("jvanz/portuguese_sentiment_analysis")
    dataset = dataset.rename_column("polarity", "label")
    dataset = dataset.map(
        lambda x: tokenizer(
            x["review_text_processed"],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        ),
        num_proc=num_proc,
        batched=True,
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) != 0)
    dataset = dataset.remove_columns(["review_text", "review_text_processed"])
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )
    # TODO remove this
    dataset["train"] = dataset["train"].select(
        range(int(dataset["train"].num_rows * 0.3))
    )
    dataset["validation"] = dataset["validation"].select(
        range(int(dataset["validation"].num_rows * 0.3))
    )
    logger.info(dataset)
    return dataset


def load_tokenizer(checkpoint):
    return BertTokenizer.from_pretrained(checkpoint)


def prepare_checkpoint_dir(config, model_name):
    config.checkpoint_dir = config.checkpoint_dir.joinpath(model_name)
    if os.path.exists(config.checkpoint_dir):
        shutil.rmtree(config.checkpoint_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)


def train_feedforward_classifier(config):
    model_name = f"feeforward_classifier_{timestamp}"
    prepare_checkpoint_dir(config, model_name)
    tokenizer = load_tokenizer(config.checkpoint)
    datasets = prepare_dataset(
        tokenizer, config.max_sequence_length, config.dataset_name
    )
    autoencoder = EncoderDecoderModel.from_pretrained(config.checkpoint)
    classifier = FeedforwardClassifier(
        autoencoder.config.encoder.hidden_size,
        config.classifier_labels,
    )
    logger.debug(autoencoder)
    logger.debug(classifier)
    if config.train:
        train(
            **vars(config),
            autoencoder=autoencoder,
            classifier=classifier,
            datasets=datasets,
            tokenizer=tokenizer,
            model_name=model_name,
        )


def fgim_attack(
    model,
    origin_data,
    target,
    ae_model,
    max_sequence_length,
    id_bos,
    id2text_sentence,
    id_to_word,
    gold_ans,
):
    """Fast Gradient Iterative Methods"""

    dis_criterion = nn.BCELoss(size_average=True)

    for epsilon in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        it = 0
        data = origin_data
        while True:
            print("epsilon:", epsilon)
            data = to_var(data.clone())  # (batch_size, seq_length, latent_size)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            output = model.forward(data)
            loss = dis_criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            data = data - epsilon * data_grad
            it += 1
            epsilon = epsilon * 0.9

            generator_id = ae_model.greedy_decode(
                data, max_len=max_sequence_length, start_id=id_bos
            )
            generator_text = id2text_sentence(generator_id[0], id_to_word)
            print("| It {:2d} | dis model pred {:5.4f} |".format(it, output[0].item()))
            print(generator_text)
            if it >= 5:
                break
    return


if __name__ == "__main__":
    config = commandline_arguments_parsing()
    logging.basicConfig(level=logging.DEBUG if config.debug else logging.INFO)
    logger.debug(config)
    train_feedforward_classifier(config)
