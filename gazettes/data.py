import csv
from urllib.parse import urlparse


def load_gazettes_csv():
    with open("data/gazettes_unix.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, dialect="unix")
        for row in reader:
            yield row


def load_gazettes_sample():
    with open("data/gazettes_sample.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, dialect="unix")
        for row in reader:
            file_name_from_url = urlparse(row["file_link"]).path.rsplit("/", 1)[-1]
            row["file_path"] = f"data/files/{file_name_from_url}"
            yield row
