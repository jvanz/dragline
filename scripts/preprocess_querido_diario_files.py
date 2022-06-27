from multiprocessing import Pool
import os

from gazettes.preprocessing import find_gazette_files, create_sentence_file


def build_destination_path(original_file: str):
    dirname = os.path.dirname(original_file)
    basefile = os.path.basename(original_file)
    return f"{dirname}/sentence_{basefile}"


def process_text_file(gazette):
    if gazette.startswith("sentence_"):
        return
    destination_file = build_destination_path(gazette)
    if os.path.exists(destination_file):
        print(f"{destination_file} already exist. Skipping...")
        return
    print(f"{gazette} -> {destination_file}")

    if os.path.exists(destination_file):
        os.remove(destination_file)
    create_sentence_file(gazette, destination_file)
    size = os.path.getsize(destination_file)
    if size <= 10:
        print(f"Destination file {destination_file} has invalid size: {size}")


if __name__ == "__main__":
    with Pool(6) as pool:
        result = pool.map(process_text_file, find_gazette_files("data/querido_diario/"))
        print(result)
