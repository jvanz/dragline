import pathlib
import os
import argparse
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, EncoderDecoderModel
from datasets import load_dataset, concatenate_datasets

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
        default=5e-5,
        help="Learning rate to be used in training steps",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
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
        default=10,
        help="Define how many step should be wait until perform evaluation",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Define how many step should be wait until log training info (e.g. losses values)",
    )

    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.joinpath(args.name)
    return args


def print_start_epoch_header(epoch: int, epochs: int):
    assert epoch >= 0
    print(f"\nEpoch [{epoch}/{epochs}]")
    print("_" * 50)


def save_models(
    autoencoder: EncoderDecoderModel,
    autoencoder_loss: float,
    #     classifier: nn.Module,
    #     classifier_loss: float,
    checkpoint_dir: str,
    epoch: int,
):
    torch.save(
        autoencoder,
        f"{checkpoint_dir}/autoencoder_{epoch}_{autoencoder_loss}.pth",
    )


#     torch.save(
#         classifier,
#         f"{checkpoint_dir}/classifier_{epoch}_{classifier_loss}.pth",
#     )


def train(
    epochs,
    autoencoder,
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
    **kw_args,
):
    print("-" * 100)
    print(f"Starting training...")
    print(f"Device used: {device}")
    train_dataloader = DataLoader(
        datasets["train"], batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        datasets["validation"], batch_size=batch_size, pin_memory=True
    )
    # evaluation_dataloader = DataLoader(
    #     datasets["evaluation"], batch_size=batch_size, pin_memory=True
    # )

    autoencoder.to(device)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    optimization_steps = train_dataloader.dataset.num_rows / train_dataloader.batch_size
    print(f"Total optimization steps: {optimization_steps}")

    current_step = 0
    patience = early_stopping_patience
    must_stop = False
    for epoch in range(epochs):
        if must_stop:
            break
        print_start_epoch_header(epoch, epochs)

        # train step
        autoencoder.train(True)
        size = train_dataloader.dataset.num_rows
        batch_size = train_dataloader.batch_size
        autoencoder_best_loss = None
        for batch in train_dataloader:
            current_step += 1
            autoencoder.zero_grad()

            # train autoencoder
            autoencoder_output = autoencoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
            autoencoder_output.loss.backward()
            autoencoder_optimizer.step()

            if current_step % logging_steps == 0:
                last_loss = autoencoder_output.loss.item()
                print(
                    f"Autoencoder loss: {last_loss:>8f}  [{current_step}/{optimization_steps}] "
                )

            autoencoder.train(False)

            if current_step % evaluation_steps == 0:
                autoencoder_loss = evaluation_step(
                    autoencoder, test_dataloader, device, tokenizer
                )

                if autoencoder_best_loss is None:
                    autoencoder_best_loss = autoencoder_loss
                    continue

                if autoencoder_best_loss - autoencoder_loss >= early_stopping_threshold:
                    patience -= 1
                else:
                    autoencoder_best_loss = autoencoder_loss
                    save_models(autoencoder, autoencoder_loss, checkpoint_dir, epoch)

                if patience == 0:
                    must_stop = True
                    break

        # save_models(autoencoder, autoencoder_loss, checkpoint_dir, epoch)
        # save_models(autoencoder, autoencoder_loss, checkpoint_dir, epoch)
    print("Training finished.")
    # evaluation_step(
    #     autoencoder,
    #     device,
    #     tokenizer,
    #     prefix_text="Evaluation losses",
    # )


def train_step(
    autoencoder,
    autoencoder_optimizer,
    dataset,
    device,
):
    current = 1
    size = dataset.dataset.num_rows
    batch_size = dataset.batch_size
    # running_loss = 0.0
    # last_loss = 0.0
    for batch in dataset:
        autoencoder.zero_grad()

        # train autoencoder
        autoencoder_output = autoencoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["input_ids"].to(device),
        )
        autoencoder_output.loss.backward()
        autoencoder_optimizer.step()

        if current % 100 == 0:
            last_loss = autoencoder_output.loss.item()
            steps = current * batch_size
            print(f"Autoencoder loss: {last_loss:>8f}  [{steps}/{size}] ")

        current += 1


def evaluation_step(
    autoencoder,
    dataset,
    device,
    tokenizer,
    prefix_text="Test losses",
):
    size = int(dataset.dataset.num_rows / dataset.batch_size)
    autoencoder_loss = 0
    current = 0
    with torch.no_grad():
        for batch in dataset:
            current += 1

            # test autoencoder
            autoencoder_output = autoencoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
            autoencoder_loss += autoencoder_output.loss.item()

    autoencoder_loss /= dataset.batch_size
    print(f"{prefix_text}:\n    Avg autoencoder loss: {autoencoder_loss:>8f}")
    return autoencoder_loss


def prepare_dataset(tokenizer, max_sequence_length, num_proc=10):
    assert tokenizer != None
    querido_diario = load_dataset(
        "pierreguillou/lener_br_finetuning_language_model", streaming=False
    )
    # querido_diario = load_dataset("jvanz/querido_diario", streaming=False)
    querido_diario = querido_diario.map(
        lambda x: {"is_legal": [1.0] * len(x["text"])}, num_proc=num_proc, batched=True
    )
    # wikipedia = load_dataset("jvanz/portuguese_wikipedia_sentences")
    # wikipedia = wikipedia.map(
    #     lambda x: {"is_legal": [0.0] * len(x["text"])}, num_proc=num_proc, batched=True
    # )

    # train_dataset = concatenate_datasets([wikipedia["train"], querido_diario["train"]])
    # test_dataset = concatenate_datasets([wikipedia["test"], querido_diario["test"]])
    # evaluation_dataset = concatenate_datasets(
    #     [wikipedia["evaluation"], querido_diario["evaluation"]]
    # )

    # querido_diario["train"] = train_dataset
    # querido_diario["test"] = test_dataset
    # querido_diario["evaluation"] = evaluation_dataset
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
            datasets=datasets,
            tokenizer=tokenizer,
        )
