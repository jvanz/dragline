import os

from transformers import (
    BertGenerationTokenizer,
    BertTokenizer,
    BertGenerationDecoder,
    BertGenerationConfig,
    EncoderDecoderModel,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader

# checkpoint = "google/bert_for_seq_generation_L-24_bbc_encoder"
checkpoint = "bert-base-uncased"
batch_size = 1000
dataset_size = 1000
max_steps = dataset_size / batch_size

model = EncoderDecoderModel.from_encoder_decoder_pretrained(checkpoint, checkpoint)
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

dataset = load_dataset("bookcorpus", streaming=True)
print(dataset)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def add_input_ids_and_labels(examples):
    return {"input_ids": examples["input_ids"], "labels": examples["input_ids"]}


# dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count())
dataset = dataset.map(tokenize_function, batched=True)
# dataset = dataset.map(add_input_ids_and_labels, batched=True, num_proc=os.cpu_count())
dataset = dataset.map(add_input_ids_and_labels, batched=True)
dataset = dataset.remove_columns("text")
dataset = dataset.with_format("torch")

print("Dataset ready.")

print("Preparing datasets for training...")
# small_train_dataset = dataset["train"].shuffle(seed=42).select(range(dataset_size))
# small_eval_dataset = dataset["train"].shuffle(seed=42).select(range(dataset_size))

small_train_dataset = dataset["train"].shuffle(seed=42).take(dataset_size)
small_eval_dataset = dataset["train"].shuffle(seed=42).take(dataset_size)
print(small_train_dataset)


training_args = TrainingArguments(output_dir="test_trainer", max_steps=dataset_size / 8)
metric = load_metric("accuracy")


def compute_metric(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metric,
)
print(trainer.get_train_dataloader())

trainer.train()
