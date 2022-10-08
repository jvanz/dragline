from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
import torch
from torch import optim, nn, Tensor
import torch.nn.functional as F
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import (
    AutoModelForPreTraining,
    BertGenerationEncoder,
    BertGenerationDecoder,
    BertGenerationConfig,
    AutoModel,
    BertConfig,
    BertLMHeadModel,
    BertModel,
)  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from datasets import load_dataset
import mlflow.pytorch


BERT_CHECKPOINT = "neuralmind/bert-base-portuguese-cased"
LEARNING_RATE = 1e-3
MAX_SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
RECONSTRUCTION_LOSS_NAME = "rec_loss"
VALIDATION_RECONSTRUCTION_LOSS_NAME = "val_rec_loss"
TEST_RECONSTRUCTION_LOSS_NAME = "test_rec_loss"
CLASSIFIER_LOSS_NAME = "classifier_loss"
EXPERIMENT_NAME = "wang_controllable_2019"
CHECKPOINT_DIR = f"checkpoints/{EXPERIMENT_NAME}"


tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, do_lower_case=False)


class AutoencoderLoss(nn.Module):
    """Implement label smoothing. From wang code"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(AutoencoderLoss, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = (
            x.data.clone()
        )  # copia x para ter um tensor da distribuição esperada igual
        true_dist.fill_(
            self.smoothing / (self.size - 2)
        )  # preenche true_dist com o mesmo valor
        true_dist.scatter_(
            1, target.data.unsqueeze(1), self.confidence
        )  # nas posições referentes aos tokens esperados, colocar o valor de self.confidence
        true_dist[
            :, self.padding_idx
        ] = 0  # na primeira posição de cada sentença colocar o self.padding_idx
        mask = torch.nonzero(
            target.data == self.padding_idx
        )  # pega a mascara de todos os tokens de padding
        if mask.dim() > 0:
            true_dist.index_fill_(
                0, mask.squeeze(), 0.0
            )  # preencher as posições de padding na true_dist com o valor 0.0
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class Classifier(nn.Module):
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

        # out = F.log_softmax(out, dim=1)
        return out  # batch_size * label_size


class AutoEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        assert checkpoint != None and len(checkpoint) > 0
        self.encoder = model = BertModel.from_pretrained(checkpoint)
        self.decoder_config = BertConfig.from_pretrained(checkpoint)
        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True
        self.decoder = BertLMHeadModel.from_pretrained(
            checkpoint, config=self.decoder_config
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        z = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        latent = self.sigmoid(z.last_hidden_state)
        latent = torch.sum(latent, dim=1)

        # Mesmo eu passando o `encoder_hidden_states` do decoder como `latent.unsqueeze(1)` (ou seja, com o shape de (batc_size, 1, dimensoes espaço latent))
        # os logits ainda vem com o shape (batch_size, max_sequence, vocab_size)
        x_hat = self.decoder(
            input_ids=batch["input_ids"], encoder_hidden_states=latent.unsqueeze(1)
        )
        probabilities = F.log_softmax(x_hat.logits, dim=-1)
        return latent, probabilities


class WangModel(pl.LightningModule):
    def __init__(
        self,
        checkpoint,
        batch_size=32,
        learning_rate=1e-3,
    ):
        super().__init__()
        assert checkpoint != None and len(checkpoint) > 0
        self.save_hyperparameters(ignore=["checkpoint"])
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.autoencoder = AutoEncoder(checkpoint)
        self.autoencoder_criterion = AutoencoderLoss(
            self.autoencoder.decoder_config.vocab_size,
            self.autoencoder.decoder_config.pad_token_id,
            smoothing=0.1,
        )

    def forward_pass(self, batch):
        z, probabilities = self.autoencoder.forward(batch)

        probabilities = probabilities.contiguous().view(-1, probabilities.size(-1))
        target_y = batch["input_ids"].contiguous().view(-1)
        tokens_count = (batch["input_ids"] != tokenizer.pad_token_id).sum().float()
        reconstruction_loss = (
            self.autoencoder_criterion(probabilities, target_y) / tokens_count
        )
        return reconstruction_loss

    def training_step(self, batch, batch_idx):
        # import pdb

        # pdb.set_trace()
        reconstruction_loss = self.forward_pass(batch)
        # self.logger.experiment.log_metric(
        #    self.logger.run_id,
        #    RECONSTRUCTION_LOSS_NAME,
        #    reconstruction_loss,
        #    step=batch_idx,
        # )
        self.log(
            RECONSTRUCTION_LOSS_NAME,
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return reconstruction_loss

    def validation_step(self, batch, batch_idx):
        reconstruction_loss = self.forward_pass(batch)
        # self.logger.experiment.log_metric(
        #    self.logger.run_id,
        #    VALIDATION_RECONSTRUCTION_LOSS_NAME,
        #    reconstruction_loss,
        #    step=batch_idx,
        # )
        self.log(
            VALIDATION_RECONSTRUCTION_LOSS_NAME,
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return reconstruction_loss

    def test_step(self, batch, batch_idx):
        reconstruction_loss = self.forward_pass(batch)
        # self.logger.experiment.log_metric(
        #    self.logger.run_id,
        #    TEST_RECONSTRUCTION_LOSS_NAME,
        #    reconstruction_loss,
        #    step=batch_idx,
        # )
        self.log(
            TEST_RECONSTRUCTION_LOSS_NAME,
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return reconstruction_loss

    def configure_optimizers(self):
        autoencoder_optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )
        # Eu nao retorno um lr_scheduler pq estou fazendo finetunning e nao preciso
        # grandes alterações nos pesos
        return autoencoder_optimizer

        # lambda1 = lambda step: self.autoencoder.decoder_config.hidden_size ** (
        #    -0.5
        # ) * min((step + 1) ** (-0.5), (step + 1) * 4000 ** (-1.5))
        # autoencoder_scheduler = optim.lr_scheduler.LambdaLR(
        #    autoencoder_optimizer, lr_lambda=[lambda1]
        # )
        # return {
        #    "optimizer": autoencoder_optimizer,
        #    "lr_scheduler": {
        #        "scheduler": autoencoder_scheduler,
        #        "interval": "step",
        #        "frequency": 1,
        #    },
        # }


class PortugueseSentimentDataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer_checkpoint, batch_size=32, max_sequence_length=60, num_proc=10
    ):
        super().__init__()
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.num_proc = num_proc

    def prepare_data(self):

        self.dataset = load_dataset("jvanz/portuguese_sentiment_analysis")
        self.dataset = self.dataset.rename_column("polarity", "label")

        self.dataset = self.dataset.map(
            lambda x: tokenizer(
                x["review_text_processed"],
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=self.max_sequence_length,
            ),
            num_proc=self.num_proc,
            batched=True,
        )
        self.dataset = self.dataset.filter(lambda x: len(x["input_ids"]) != 0)
        self.dataset = self.dataset.remove_columns(
            ["review_text", "review_text_processed"]
        )
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
        )


def main(args):
    model = WangModel(BERT_CHECKPOINT, batch_size=BATCH_SIZE)

    mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME)
    trainer = pl.trainer.trainer.Trainer.from_argparse_args(
        args,
        callbacks=[
            EarlyStopping(monitor=RECONSTRUCTION_LOSS_NAME, patience=5),
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                filename=f"{{epoch}}-{{step}}-{{{RECONSTRUCTION_LOSS_NAME}:.6f}}",
                save_top_k=3,
                monitor=RECONSTRUCTION_LOSS_NAME,
                save_last=True,
                auto_insert_metric_name=True,
                save_weights_only=False,
                every_n_train_steps=5000,
            ),
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=3),
        ],
        logger=mlf_logger,
    )

    datamodule = PortugueseSentimentDataModule(BERT_CHECKPOINT, batch_size=BATCH_SIZE)

    mlflow.pytorch.autolog()

    print(f"Resuming from: {args.checkpoint_path}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint_path)
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="The checkpoint to resume the training from.",
    )
    parser = pl.trainer.trainer.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
