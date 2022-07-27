import os

from transformers import (
    BertGenerationTokenizer,
    BertTokenizer,
    BertGenerationDecoder,
    BertGenerationConfig,
    EncoderDecoderModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
import numpy as np

model_name = "jvanz/querido_diario_autoencoder"
checkpoint = "neuralmind/bert-base-portuguese-cased"
MAX_SEQUENCE_LENGTH = 20
BATCH_SIZE = 128
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.01
OUTPUT_DIR = f"checkpoints/{model_name}"
RESUME_TRAIN = False
PARTIAL_DATASET = 0.1
EVAL_STEPS = 2000
NO_CUDA = True


os.makedirs(OUTPUT_DIR, exist_ok=True)

model = EncoderDecoderModel.from_encoder_decoder_pretrained(checkpoint, checkpoint)
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

dataset = load_dataset("jvanz/querido_diario", streaming=False)
dataset = dataset.remove_columns(
    [
        "input_ids",
        "single_letter_tokens_count",
        "total_tokens_count",
    ]
)
print(dataset)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )


def add_input_ids_and_labels(examples):
    return {"input_ids": examples["input_ids"], "labels": examples["input_ids"]}


dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count())
dataset = dataset.map(add_input_ids_and_labels, batched=True, num_proc=os.cpu_count())
dataset = dataset.remove_columns("text")
dataset = dataset.with_format("torch")

print(dataset)
print("Dataset ready.")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="steps",
    eval_accumulation_steps=1,
    save_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    num_train_epochs=EPOCHS,
    log_level="debug",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    save_total_limit=5,
    push_to_hub=True,
    hub_model_id=model_name,
    no_cuda=NO_CUDA,
)

early_stop_callback = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["evaluation"],
    callbacks=[early_stop_callback],
    tokenizer=tokenizer,
)

trainer.train(resume_from_checkpoint=RESUME_TRAIN)
