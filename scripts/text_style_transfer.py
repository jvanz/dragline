import pathlib
import os
import argparse
import logging
import shutil

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
        self.model = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_size),
            nn.Sigmoid(),
        )

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
        "--model-name",
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
    args.checkpoint_dir = args.checkpoint_dir.joinpath(args.model_name)
    return args


def print_start_epoch_header(epoch: int, epochs: int):
    assert epoch >= 0
    print(f"\nEpoch [{epoch}/{epochs}]")
    print("_" * 50)


def save_models(
    autoencoder: EncoderDecoderModel,
    autoencoder_loss: float,
    classifier: nn.Module,
    classifier_loss: float,
    checkpoint_dir: str,
    step: int,
    model_name: str,
):
    torch.save(
        autoencoder,
        f"{checkpoint_dir}/autoencoder_{model_name}_{step}_{autoencoder_loss}.pth",
    )
    torch.save(
        classifier,
        f"{checkpoint_dir}/classifier_{model_name}_{step}_{classifier_loss}.pth",
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
    train_dataloader = DataLoader(
        datasets["train"], batch_size=batch_size, pin_memory=True
    )
    evaluation_dataloader = DataLoader(
        datasets["test"], batch_size=batch_size, pin_memory=True
    )

    autoencoder.to(device)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    classifier.to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    classifier_loss_fn = nn.BCELoss()

    sigmoid = nn.Sigmoid()

    optimization_steps = (
        int(train_dataloader.dataset.num_rows / train_dataloader.batch_size) * epochs
    )
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
        classifier.train(True)
        size = train_dataloader.dataset.num_rows
        batch_size = train_dataloader.batch_size
        autoencoder_best_loss = None
        classifier_best_loss = None
        for batch in train_dataloader:
            current_step += 1
            autoencoder.zero_grad()
            classifier.zero_grad()

            # train autoencoder
            autoencoder_output = autoencoder(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
            autoencoder_latent_space = torch.sum(
                sigmoid(autoencoder_output.encoder_last_hidden_state).detach(), dim=1
            )
            autoencoder_output.loss.backward()
            autoencoder_optimizer.step()

            classifier_output = classifier(autoencoder_latent_space)
            classifier_loss = classifier_loss_fn(
                classifier_output.flatten(), batch["label"].to(device)
            )
            classifier_loss.backward()
            classifier_optimizer.step()

            if current_step % logging_steps == 0:
                autoencoder_last_loss = autoencoder_output.loss.item()
                classifier_last_loss = classifier_loss.item()
                print(
                    f"Autoencoder loss: {autoencoder_last_loss:>8f}, Classifier loss: {classifier_last_loss}  [{current_step}/{optimization_steps}] "
                )

            autoencoder.train(False)
            classifier.train(False)

            if current_step % evaluation_steps == 0:
                current_evaluation_step = 0
                autoencoder_loss = 0.0
                classifier_loss = 0.0
                with torch.no_grad():
                    for batch in evaluation_dataloader:
                        current_evaluation_step += 1

                        # test autoencoder
                        autoencoder_output = autoencoder(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["input_ids"].to(device),
                        )
                        autoencoder_loss += autoencoder_output.loss.item()
                        autoencoder_latent_space = torch.sum(
                            sigmoid(
                                autoencoder_output.encoder_last_hidden_state
                            ).detach(),
                            dim=1,
                        )

                        classifier_output = classifier(autoencoder_latent_space)
                        classifier_last_loss = classifier_loss_fn(
                            classifier_output.flatten(), batch["label"].to(device)
                        )
                        classifier_loss += classifier_last_loss.item()

                autoencoder_loss /= current_evaluation_step
                classifier_loss /= current_evaluation_step
                print(
                    f"Evaluation:\n    Avg autoencoder loss: {autoencoder_loss:>8f}, Avg classifier loss: {classifier_loss:>8f}"
                )
                save_models(
                    autoencoder,
                    autoencoder_loss,
                    classifier,
                    classifier_loss,
                    checkpoint_dir,
                    current_step,
                    model_name,
                )

                # if autoencoder_best_loss is None:
                #     autoencoder_best_loss = autoencoder_loss
                #     classifier_best_loss = classifier_loss
                #     continue

                # if (
                #     autoencoder_best_loss - autoencoder_loss
                # ) < early_stopping_threshold and (
                #     classifier_best_loss - classifier_loss
                # ) < early_stopping_threshold:
                #     patience -= 1
                # else:
                #     autoencoder_best_loss = autoencoder_loss
                #     classifier_best_loss = classifier_loss
                #     save_models(
                #         autoencoder,
                #         autoencoder_loss,
                #         classifier,
                #         classifier_loss,
                #         checkpoint_dir,
                #         current_step,
                #         model_name,
                #     )

                # if patience == 0:
                #     must_stop = True
                #     print(
                #         f"Stopping training earlier. Loss value did not improved more then {early_stopping_threshold} since last evaluation."
                #     )
                #     break

    print("Training finished.")


def prepare_dataset(tokenizer, max_sequence_length, dataset_name: str, num_proc=10):
    if dataset_name == "sentiment":
        return prepare_sentiment_dataset(tokenizer, max_sequence_length, num_proc)
    return prepare_legal_dataset(tokenizer, max_sequence_length, num_proc)


def prepare_sentiment_dataset(tokenizer, max_sequence_length, num_proc):
    dataset = load_dataset("jvanz/portuguese_sentiment_analysis", split="train")
    dataset = dataset.rename_column("polarity", "label")
    dataset = dataset.map(
        lambda x: tokenizer(
            x["review_text"],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        ),
        num_proc=num_proc,
        batched=True,
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) != 0)
    dataset = dataset.remove_columns(["review_text"])
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )
    logger.info(dataset)
    return dataset


def prepare_legal_dataset(tokenizer, max_sequence_length, num_proc):
    assert tokenizer != None
    legal_text = load_dataset(
        "pierreguillou/lener_br_finetuning_language_model", streaming=False
    )
    # querido_diario = load_dataset("jvanz/querido_diario", streaming=False)
    legal_text = legal_text.map(
        lambda x: {"is_legal": [1.0] * len(x["text"])}, num_proc=num_proc, batched=True
    )
    wikipedia = load_dataset("jvanz/portuguese_wikipedia_sentences")
    wikipedia = wikipedia.map(
        lambda x: {"is_legal": [0.0] * len(x["text"])}, num_proc=num_proc, batched=True
    )

    train_dataset = concatenate_datasets([wikipedia["train"], legal_text["train"]])
    # test_dataset = concatenate_datasets([wikipedia["test"], querido_diario["test"]])
    evaluation_dataset = concatenate_datasets(
        [wikipedia["evaluation"], legal_text["validation"]]
    )

    legal_text["train"] = train_dataset
    # querido_diario["test"] = test_dataset
    legal_text["evaluation"] = evaluation_dataset
    legal_text = legal_text.map(
        lambda x: tokenizer(
            x["text"],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        ),
        num_proc=num_proc,
        batched=True,
    )
    legal_text.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "is_legal"],
    )
    legal_text = legal_text.shuffle(seed=42)
    logger.info(legal_text)
    return legal_text


def load_tokenizer(checkpoint):
    return BertTokenizer.from_pretrained(checkpoint)


def create_models(
    checkpoint: str, classifier_labels: int, tokenizer: BertTokenizer, config
):
    assert checkpoint != None and len(checkpoint) > 0
    assert classifier_labels > 0
    autoencoder = EncoderDecoderModel.from_encoder_decoder_pretrained(
        checkpoint, checkpoint
    )
    autoencoder.config.decoder_start_token_id = tokenizer.cls_token_id
    autoencoder.config.pad_token_id = tokenizer.cls_token_id
    autoencoder.config.max_length = config.max_sequence_length
    classifier = Classifier(autoencoder.config.encoder.hidden_size, classifier_labels)
    logger.debug(autoencoder)
    logger.debug(classifier)
    return autoencoder, classifier


if __name__ == "__main__":
    config = commandline_arguments_parsing()
    logging.basicConfig(level=logging.DEBUG if config.debug else logging.INFO)
    logger.debug(config)
    if os.path.exists(config.checkpoint_dir):
        shutil.rmtree(config.checkpoint_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer = load_tokenizer(config.checkpoint)
    datasets = prepare_dataset(
        tokenizer, config.max_sequence_length, config.dataset_name
    )
    autoencoder, classifier = create_models(
        config.checkpoint, config.classifier_labels, tokenizer, config
    )
    if config.train:
        train(
            **vars(config),
            autoencoder=autoencoder,
            classifier=classifier,
            datasets=datasets,
            tokenizer=tokenizer,
        )
