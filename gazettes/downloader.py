from urllib.parse import urlparse
from multiprocessing import Pool

import requests


def _download_file(url: str, destination: str, file_name: str) -> None:
    """
    TODO
    """
    with open(f"{destination}/{file_name}", "wb") as destination_file:
        result = requests.get(url, stream=True)
        for chunk in result.iter_content(chunk_size=512):
            destination_file.write(chunk)


def _get_file_name_from_url(file_url: str) -> str:
    """
    TODO
    """
    print(urlparse(file_url).path)
    return urlparse(file_url).path.rsplit("/", 1)[-1]


def _build_parallel_function_arguments(files_urls, destination):
    """
    TODO
    """
    for url in files_urls:
        yield (url, destination, _get_file_name_from_url(url))


def download_files(files_urls, destination, concurrent_process=4, timeout=None):
    """
    TODO
    """
    arguments = _build_parallel_function_arguments(files_urls, destination)
    with Pool(concurrent_process) as pool:
        result = pool.starmap_async(_download_file, arguments)
        result.get(timeout)
