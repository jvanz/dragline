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
    BertForPreTraining,
    EncoderDecoderModel,
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


#
class AutoEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        assert checkpoint != None and len(checkpoint) > 0
        self.encoder = BertModel.from_pretrained(checkpoint)
        self.decoder = BertLMHeadModel.from_pretrained(checkpoint)
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

        return latent, x_hat.logits, F.log_softmax(x_hat.logits, dim=-1)

    def generate(self, batch, bos_token_id, pad_token_id, max_sequence_length):
        z = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        latent = self.sigmoid(z.last_hidden_state)
        latent = torch.sum(latent, dim=1)

        y = torch.zeros(batch["input_ids"].size(0), 1).fill_(bos_token_id).long()
        for i in range(max_sequence_length - 1):
            x_hat = self.decoder(input_ids=y, encoder_hidden_states=latent.unsqueeze(1))
            probabilities = F.log_softmax(x_hat.logits, dim=-1)[:, -1, :]
            _, next_word = torch.max(probabilities, dim=-1)
            y = torch.cat([y, next_word.unsqueeze(1)], dim=1)
        return y


def main():
    checkpoint = torch.load(
        "./epoch=12-step=720000-rec_loss=2.015147.ckpt", map_location="cpu"
    )
    autoencoder = AutoEncoder(BERT_CHECKPOINT)
    encdec = EncoderDecoderModel.from_pretrained("jvanz/querido_diario_autoencoder")

    # Como o checkpoint foi treinado em um modulo lightning, preciso ajustar as keys.
    # state_dict = {}
    # for key in checkpoint["state_dict"]:
    #     state_dict[key[len("autoencoder.") :]] = checkpoint["state_dict"][key]

    # import pdb

    # pdb.set_trace()

    # print(autoencoder.load_state_dict(state_dict, strict=False))

    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, do_lower_case=False)

    batch = tokenizer(
        ["Quero ver se o meu modelo está funcionando"],
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    batch["input_ids"] = torch.Tensor(batch["input_ids"]).long()
    batch["token_type_ids"] = torch.Tensor(batch["token_type_ids"]).long()
    batch["attention_mask"] = torch.Tensor(batch["attention_mask"]).long()

    autoencoder.encoder = encdec.encoder
    autoencoder.decoder = encdec.decoder

    outputs = autoencoder.generate(
        batch,
        bos_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    print(outputs)
    print(outputs.size())
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
