import json
from multiprocessing import Pool
import os

from requests import put

from gazettes.data import load_gazettes_sample


def extract_file_metadata(file_path: str, destination: str):
    """Send a request to the Apache Tika to extract file's metadata.

    Send a request to the Apache Tika to extract file's metadata and writes the
    response in the destination file.

    :returns: dict with the file's metadata
    """
    with open(file_path, "rb") as file:
        response = put(
            "http://localhost:9998/meta",
            headers={"Accept": "application/json"},
            data=file,
        )
        if response.status_code == 200:
            metadata = response.json()
            with open(destination, "w") as destination_file:
                destination_file.write(json.dumps(metadata))
            return metadata
        else:
            raise Exception(
                f"Cannot extract metadata from file '{file_path}': {response.text}"
            )
        raise Exception(f"Cannot extract metadata from file '{file_path}'")


def extract_file_text(file_path: str, content_type: str, destination: str):
    """Send a request to the Apache Tika to extract file's content.

    Send a request to the Apache Tika to extract file's content and writes the
    response in the destination file.

    """
    with open(file_path, "rb") as file:
        response = put(
            "http://localhost:9998/tika",
            headers={"Content-type": content_type, "Accept": "text/xml"},
            data=file,
        )
        if response.status_code == 200:
            with open(destination, "wb") as destination_file:
                for chunk in response.iter_content(chunk_size=512):
                    destination_file.write(chunk)
        else:
            raise Exception(
                f"Cannot extract text from file '{file_path}': {response.text}"
            )


def get_file_to_store_extracted_text(file_path: str):
    """Generates the file name used to store the file's content

    :file_path: original file path
    :returns: file path used to store the file's content
    """
    file_name = os.path.basename(os.path.splitext(file_path)[0])
    return f"data/files/{file_name}.xml"


def get_file_to_store_extracted_metadata(file_path: str):
    """Generates the file name used to store the metadata

    :file_path: original file path
    :returns: file path used to store the metadata information
    """
    file_name = os.path.basename(os.path.splitext(file_path)[0])
    return f"data/files/{file_name}.json"


def extract_file_info(gazette):
    """TODO: Function called by the processes in the pool to process a gazette
    file

    :gazette: dict with the gazette info
    """
    text_extracted_metadata_file_destination = get_file_to_store_extracted_metadata(
        gazette["file_path"]
    )
    metadata = extract_file_metadata(
        gazette["file_path"], text_extracted_metadata_file_destination
    )
    text_extracted_file_destination = get_file_to_store_extracted_text(
        gazette["file_path"]
    )
    extract_file_text(
        gazette["file_path"], metadata["Content-Type"], text_extracted_file_destination,
    )


def _build_parallel_function_arguments(gazettes):
    """
    Build the arguments to be passed in the function executed by the processes
    in the pool
    """
    for gazette in gazettes:
        yield (gazette,)


def extract_files_info(gazettes, concurrent_process=4, timeout=None):
    """Extract the file's content and metadata

    Run a pool of multiple processes to extract the file content and its
    metadata. This is done sending requests to a Apache Tika  server running
    locally on port 9998.
    """
    with Pool(concurrent_process) as pool:
        arguments = _build_parallel_function_arguments(gazettes)
        result = pool.starmap_async(extract_file_info, arguments)
        result.get(timeout)
