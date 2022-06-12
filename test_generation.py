import os

from transformers import (
    BertGenerationTokenizer,
    BertTokenizer,
    BertGenerationDecoder,
    BertGenerationConfig,
    EncoderDecoderModel,
)
from datasets import load_dataset

checkpoint = "neuralmind/bert-base-portuguese-cased"
#checkpoint = "data/lenerbr-generation/checkpoint-8000/"
MAX_SEQUENCE_LENGTH = 20


model = EncoderDecoderModel.from_pretrained(checkpoint)
tokenizer = BertTokenizer.from_pretrained("pierreguillou/bert-base-cased-pt-lenerbr")
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

frase_test = (
    "Essa é a primeira frase que vou tentar gerar utilizando o meu modelo treinado."
)
input_ids = tokenizer(
    frase_test, add_special_tokens=False, return_tensors="pt"
).input_ids
print(input_ids)
outputs = model.generate(input_ids)
print(outputs)
for sequence in outputs:
    frase_gerada = tokenizer.decode(sequence)
    print(f"Frase original: {frase_test}. Frase gerada: {frase_gerada}")


dataset = load_dataset(
    "pierreguillou/lener_br_finetuning_language_model", streaming=False
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

dataset = dataset["validation"].shuffle().select(range(50))
dataset = dataset.with_format("pt")
print(dataset)
print(dataset["input_ids"])
outputs = model.generate(dataset["input_ids"])
print(outputs)
originais = tokenizer.batch_decode(dataset["input_ids"])
print(originais)
sentences = tokenizer.batch_decode(outputs)
print(sentences)

print("-------------------------------------------")
for original, generated in zip(originais, sentences):
    print(f"{original} ---> {generated}")
