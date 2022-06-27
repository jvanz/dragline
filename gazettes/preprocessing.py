from collections import Counter
import re
import os
from datetime import datetime
import logging

import spacy

REMOVE_WHITESPACES_REGEX = r"[\t ]{2,}"
REMOVE_CONSECUTIVE_EMPTY_LINES_REGEX = r"^\n{2,}"
REMOVE_NEW_LINES_REGEX = r"\s+"
REMOVE_LINES_WITH_PUNCTUATION_ONLY = (
    r" […” !\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`\{|\}~]+ |\.{2,} ?"
)
REMOVE_ART_LINE_PREFIX = r"^Art\. ?\d+\.?º? ?"
REMOVE_INCISO_LINE_PREFIX = r"^§ \d+\.?º? ?"
MINIMUM_SENTENCE_TOKEN_COUNT = 8
MATCH_ART_SUFFIX = r".*Art\.? \d+\.$"
MATCH_SEC_SUFFIX = r".*Sec\.?$"

SENTENCE_FILE_PREFIX = "sentence_"


def remove_word_with_duplicate_letters(text: str):
    """
    AABBCC DDEEFF
    """
    for duplicate_char in find_duplicates_chars(text):
        regex = r"" + duplicate_char + "{2,}"
        text = re.sub(regex, duplicate_char, text)
    return text


def remove_consecutive_empty_lines(text: str):
    return re.sub(REMOVE_CONSECUTIVE_EMPTY_LINES_REGEX, "\n", text)


def remove_new_line_char(text: str):
    return re.sub(REMOVE_NEW_LINES_REGEX, " ", text).strip()


def find_duplicates_chars(text: str):
    char_count = Counter(text)
    return [char for char, count in char_count.items() if count > 1 and char != " "]


def remove_duplicate_whitespaces(text: str):
    return re.sub(REMOVE_WHITESPACES_REGEX, " ", text)


def remove_line_with_punctuation_only(text: str):
    return re.sub(REMOVE_LINES_WITH_PUNCTUATION_ONLY, " ", text)


def remove_special_line_prefix(text: str):
    text = re.sub(REMOVE_ART_LINE_PREFIX, "", text, flags=re.MULTILINE)
    return re.sub(REMOVE_INCISO_LINE_PREFIX, "", text, flags=re.MULTILINE)


def remove_special_quotes(text: str):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return text


def _preprocess_text(text: str):
    text = remove_special_quotes(text)
    text = remove_special_line_prefix(text)
    text = remove_new_line_char(text)
    return remove_line_with_punctuation_only(text)


def preprocess_gazette_txt_file(text_file, destination_file):
    with open(text_file, "r") as original_file:
        with open(destination_file, "w") as destination:
            destination.write(_preprocess_text(original_file.read()))


def _should_sentence_be_merged(sentence):
    return (
        sentence.text.strip().endswith("inc.")
        or re.fullmatch(MATCH_ART_SUFFIX, sentence.text.strip()) is not None
        or re.fullmatch(MATCH_SEC_SUFFIX, sentence.text.strip()) is not None
    )


def _should_skip_sentence(sentence):
    return len(sentence) <= MINIMUM_SENTENCE_TOKEN_COUNT


def sentence_segmentation(text: str):
    nlp = spacy.load("pt_core_news_lg")
    nlp.disable_pipe("parser")
    nlp.enable_pipe("senter")
    nlp.max_length = 30000000
    merge_sentence = ""
    for sentence in nlp(text).sents:
        if _should_sentence_be_merged(sentence):
            merge_sentence += sentence.text
            continue

        if merge_sentence:
            yield merge_sentence + sentence.text
            merge_sentence = ""
            continue

        if _should_skip_sentence(sentence):
            continue

        yield sentence.text


def create_sentence_file(text_file: str, destination_file: str):
    with open(text_file, "r") as original_file:
        with open(destination_file, "w") as destination:
            for sentence in sentence_segmentation(
                _preprocess_text(original_file.read())
            ):
                destination.write(f"{sentence}\n")


def _is_directory_date(directory: str, since: str):
    if since is None:
        return True
    try:
        DATE_FORMAT = "%Y-%m-%d"
        directory_date = datetime.strptime(os.path.basename(directory), DATE_FORMAT)
        date = datetime.strptime(since, DATE_FORMAT)
        return directory_date >= date
    except Exception as e:
        logging.getLogger().debug(f"Cannot parse date from directory: {directory}")
    return False


def find_gazette_files(root_dir: str, since: str = None):
    for dirpath, _, files in os.walk(root_dir):
        if not _is_directory_date(dirpath, since=since):
            continue
        for filee in files:
            if filee.endswith(".txt") and not filee.startswith(SENTENCE_FILE_PREFIX):
                yield f"{dirpath}/{filee}"


def find_sentence_gazette_files(root_dir: str, since: str = None):
    for dirpath, _, files in os.walk(root_dir):
        if not _is_directory_date(dirpath, since=since):
            continue
        for filee in files:
            if filee.endswith(".txt") and filee.startswith(SENTENCE_FILE_PREFIX):
                yield f"{dirpath}/{filee}"
