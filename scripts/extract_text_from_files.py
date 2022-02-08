from gazettes.text_extraction import extract_files_info
from gazettes.data import load_gazettes_sample


def main():
    gazettes = list(load_gazettes_sample())
    extract_files_info(gazettes)


if __name__ == "__main__":
    main()
