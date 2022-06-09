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

# checkpoint = "google/bert_for_seq_generation_L-24_bbc_encoder"
checkpoint = "bert-base-uncased"
train_dataset_size = 100000
eval_dataset_size = 1000
MAX_SEQUENCE_LENGTH = 20
BATCH_SIZE = 32
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 500
EARLY_STOPPING_THRESHOLD = 0.01
OUTPUT_DIR = "/data/test_trainer"
RESUME_TRAIN = True

model = EncoderDecoderModel.from_encoder_decoder_pretrained(checkpoint, checkpoint)
tokenizer = BertTokenizer.from_pretrained(checkpoint)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

dataset = load_dataset("bookcorpus", streaming=False, split="train")
dataset = dataset.train_test_split(shuffle=True)

print(dataset)
# small_train_dataset = (
#     dataset["train"].shuffle(seed=42).select(range(train_dataset_size))
# )
# small_eval_dataset = dataset["train"].shuffle(seed=42).select(range(eval_dataset_size))


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )


def add_input_ids_and_labels(examples):
    return {"input_ids": examples["input_ids"], "labels": examples["input_ids"]}


dataset = dataset.map(
    tokenize_function, batched=True, num_proc=os.cpu_count()
)
# dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.map(
    add_input_ids_and_labels, batched=True, num_proc=os.cpu_count()
)
# dataset = dataset.map(add_input_ids_and_labels, batched=True)
dataset = dataset.remove_columns("text")
dataset = dataset.with_format("torch")
#
# small_eval_dataset = small_eval_dataset.map(
#     tokenize_function, batched=True, num_proc=os.cpu_count()
# )
# # small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)
# small_eval_dataset = small_eval_dataset.map(
#     add_input_ids_and_labels, batched=True, num_proc=os.cpu_count()
# )
# # small_eval_dataset = small_eval_dataset.map(add_input_ids_and_labels, batched=True)
# small_eval_dataset = small_eval_dataset.remove_columns("text")
# small_eval_dataset = small_eval_dataset.with_format("torch")

print("Dataset ready.")
# print(small_train_dataset)
# print(small_eval_dataset)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_accumulation_steps=1,
    save_strategy="steps",
    eval_steps=500,
    num_train_epochs=EPOCHS,
    log_level="debug",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    save_total_limit=5,
)
# metric = load_metric("mse", "multilist")


# def compute_metric(eval_pred):
#     logits, labels = eval_pred
#     logits = logits[0]
#
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#     # import pdb
#     # pdb.set_trace()
#     # metric_value = metric.compute(predictions=predictions, references=labels)
#     # metric_value["eval_loss"] = metric_value["mse"]
#     # return metric_value


early_stop_callback = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    # compute_metrics=compute_metric,
    callbacks=[early_stop_callback],
)
print(trainer.get_train_dataloader())

trainer.train(resume_from_checkpoint=RESUME_TRAIN)
