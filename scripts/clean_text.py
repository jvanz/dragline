from gazettes.data import sample_gazettes_texts


def main():
    for gazette, text in sample_gazettes_texts(force_clean=True):
        print(f"{gazette['file_path']}: {text}")


if __name__ == "__main__":
    main()
