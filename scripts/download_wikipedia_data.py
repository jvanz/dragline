import csv

from datasets import load_dataset


DATA_DIR = os.environ("DATA_DIR", "data")


def has_no_minimum_words(sentence):
    return len(sentence.split(" ")) <= 3


def has_invalid_char(sentence):
    return "|" in sentence


def sentence_segmentation(sample):
    sentences = []
    for text in sample["text"]:
        for sentence in text.split("."):
            sentence = sentence.replace("\n", " ").strip()
            if has_no_minimum_words(sentence) or has_invalid_char(sentence):
                continue
            sentences.append(sentence)
    sample = {"text": sentences}
    return sample


def main():
    dataset_name = "wikipedia"
    dataset_language = "pt"
    dataset_date = "20220220"
    dataset = load_dataset(
        "wikipedia",
        language=dataset_language,
        date=dataset_date,
        beam_runner="DirectRunner",
    )
    print(dataset)
    dataset = dataset.map(
        sentence_segmentation, batched=True, remove_columns="title", num_proc=4
    )
    print(dataset)
    dataset["train"].to_csv(
        f"{DATA_DIR}/{dataset_name}_{dataset_date}_{dataset_language}.csv",
        num_proc=6,
        quoting=csv.QUOTE_ALL,
        index=False,
    )


if __name__ == "__main__":
    main()
