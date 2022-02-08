import csv
import random

from gazettes.downloader import download_files
from gazettes.data import load_gazettes_csv


def main():
    data = list(load_gazettes_csv())
    sample = random.sample(data, 5000)

    with open("data/gazettes_sample.csv", "w") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            dialect="unix",
            fieldnames=[
                "autopublicacao",
                "category",
                "date",
                "entity",
                "file_link",
                "files",
                "power",
                "scraped_at",
                "title",
            ],
        )
        writer.writeheader()
        for row in sample:
            writer.writerow(row)

    urls = [row["file_link"] for row in sample]
    download_files(urls, "data/files")


if __name__ == "__main__":
    main()
