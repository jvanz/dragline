from datasets import load_dataset, Dataset
from transformers import BertTokenizer

from gazettes.preprocessing import find_sentence_gazette_files

DATASET_NAME = "jvanz/querido_diario"
BERT_PTBR_CHECKPOINT = "neuralmind/bert-base-portuguese-cased"
REPLACEMENT_CHAR_NUMBER = chr(65533)
NUM_PROC = 10


def read_files_and_publish():

    tokenizer = BertTokenizer.from_pretrained(BERT_PTBR_CHECKPOINT)

    files = list(find_sentence_gazette_files("data/querido_diario/"))
    dataset = load_dataset("text", data_files=files)
    print(dataset)
    print("-" * 100)

    def tokenizer_sentences(examples):
        tokenizer_output = tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        single_letter_token = []
        total_tokens = []
        for tokens in tokenizer_output["input_ids"]:
            single_letter_token_count = 0
            total_tokens.append(len(tokens))
            for token in tokens:
                if len(tokenizer.ids_to_tokens[token]) == 1:
                    single_letter_token_count += 1
            single_letter_token.append(single_letter_token_count)

        return {
            "text": examples["text"],
            "input_ids": tokenizer_output["input_ids"],
            "single_letter_tokens_count": single_letter_token,
            "total_tokens_count": total_tokens,
        }

    dataset = dataset.filter(
        lambda x: REPLACEMENT_CHAR_NUMBER not in x["text"], num_proc=NUM_PROC
    )
    print(dataset)
    print(dataset["train"][:10])
    print("-" * 100)
    dataset = dataset.map(tokenizer_sentences, batched=True, num_proc=NUM_PROC)

    def filter_sentence_with_too_much_single_letter_tokens(example):
        filter_list = []
        for total_tokens, single_letter_tokens in zip(
            example["total_tokens_count"], example["single_letter_tokens_count"]
        ):
            filter_list.append(single_letter_tokens < total_tokens * 0.2)
        return filter_list

    print(dataset)
    print(dataset["train"][:10])
    print("-" * 100)
    dataset = dataset.filter(
        filter_sentence_with_too_much_single_letter_tokens,
        batched=True,
        num_proc=NUM_PROC,
    )
    print(dataset)
    print(dataset["train"][:10])
    print("-" * 100)
    dataset = dataset.remove_columns(
        [
            "input_ids",
            "single_letter_tokens_count",
            "total_tokens_count",
        ]
    )

    dataset = dataset["train"].train_test_split(shuffle=True)
    test = dataset["test"]
    test = test.train_test_split(test_size=0.5)
    dataset["test"] = test["train"]
    dataset["evaluation"] = test["test"]
    print(dataset)
    print("-" * 100)

    dataset.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    read_files_and_publish()
