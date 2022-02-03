from gazettes.downloader import download_files
from gazettes import load_gazettes_csv


def main():
    urls = []
    for gazette in load_gazettes_csv():
        urls.append(gazette["file_link"])
        if len(urls) == 10:
            break

    download_files(urls, "data/files")


if __name__ == "__main__":
    main()
