"""Read text from a file with different formats"""

from __future__ import annotations

import mimetypes
import os
import re
import tempfile

from importlib import import_module
from pathlib import Path
from typing import Any, List
from urllib.parse import urlparse

import requests

from bs4 import BeautifulSoup
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.pdf import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document


def cleanup_text(text_in: str) -> str:
    """Cleanup input text from multple spaces and newlines"""
    new_text = re.sub(r"\n\n+", "\n\n", text_in)
    new_text = new_text.replace("\x00", " ")
    new_text = re.sub(r"\.\.\.+", "...", new_text)

    return " ".join(new_text.strip().split(" "))


def read_pdf_file(
    file_path: str,
    page_from: int = None,  # from page, used in PDF
    page_to: int = None,  # to page, used in PDF
    **loader_kwargs: Any,
) -> str:
    """
    Load PDF file and return its textual content
    """
    docs = load_documents_pdf(file_path=file_path, **loader_kwargs)
    if page_from is not None:
        docs = [x for x in docs if x.metadata["page"] >= (page_from - 1)]
    if page_to is not None:
        docs = [x for x in docs if x.metadata["page"] <= (page_to - 1)]

    return " ".join([cleanup_text(item.page_content) for item in docs]).strip()


def load_documents_pdf(file_path: str, **kwargs: Any) -> list[Document]:
    """
    Load documents from PDF file, first try to extract text with PyPDF,
    then use Unstructured and OCR to read
    """
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    # check if `ocr` argument is passed and not falsy
    if "ocr" not in kwargs or not kwargs["ocr"]:
        return docs

    # try with OCR if extracted text is too short - optional 'min_text_len' parameter
    min_text_len = kwargs["min_text_len"] if "min_text_len" in kwargs else 100
    text = " ".join([cleanup_text(item.page_content) for item in docs]).strip()
    if len(text) > min_text_len:
        return docs

    ocr_languages = kwargs["ocr_languages"] if "ocr_languages" in kwargs else None
    loader = UnstructuredPDFLoader(
        file_path=file_path,
        mode="elements",
        strategy="ocr_only",
        ocr_languages=ocr_languages,
    )

    return loader.load()


def read_txt_file(
    file_path: str,
) -> str:
    """
    Load TXT file and return its content
    """
    with open(file_path) as file:  # pylint: disable=unspecified-encoding
        text = file.read()

    return cleanup_text(text).strip()


def read_csv_file(
    file_path: str,
) -> str:
    """
    Load CSV file and return its content
    """
    docs = load_documents_csv(file_path=file_path)

    return " ".join([cleanup_text(item.page_content) for item in docs]).strip()


def read(
    file_path: str,
    **loader_kwargs: Any,
) -> str:
    """
    Load text from a file, see `MIME_TYPES_READERS` for supported content types
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    mtype = mimetypes.guess_type(file_path)[0]
    if mtype not in MIME_TYPES_LOADERS:
        raise ValueError(f'Unsupported file content type "{mtype}"')

    read_function = MIME_TYPES_READERS[mtype]

    return read_function(file_path, **loader_kwargs)


def read_html_url(
    url: str,
    **loader_kwargs: Any,
) -> str:
    """
    Load text from HTML content of web page URL
    """
    parsed_url = urlparse(url)
    remote = True if parsed_url.scheme not in ["file", ""] else False
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    if remote:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
    else:
        text = Path(url).read_text()

    with open(temp_file.name, "w") as file:  # pylint: disable=missing-timeout
        content = filter_html(
            html=text, selector=loader_kwargs.pop("selector", ""), callback=loader_kwargs.pop("callback", "")
        )
        file.write(content)
    loader = BSHTMLLoader(file_path=temp_file.name, **loader_kwargs)
    docs = loader.load()
    os.unlink(temp_file.name)

    return " ".join([cleanup_text(item.page_content) for item in docs]).strip()


def filter_html(html: str, selector: str = "", callback: str = "") -> str:
    """Filter HTML content using callbacks and selector rule, using BeautifulSoup"""
    if callback:
        module_name, func_name = callback.rsplit(".", 1)
        try:
            module = import_module(module_name)
            html = getattr(module, func_name)(html)  # pylint: disable=not-callable
        except (ImportError, AttributeError):
            raise ValueError(f'Callback "{callback}" not found')
    if not selector:
        return html
    soup = BeautifulSoup(html, features="lxml")
    tags = soup.select(selector=selector)

    return " ".join([tag.decode_contents() for tag in tags])


def load_documents(file_path: str, **kwargs: Any) -> list[Document]:
    """Load documents from a file or folder path"""
    mtype = mimetypes.guess_type(file_path)[0]
    if mtype not in MIME_TYPES_LOADERS:
        raise ValueError(f'Unsupported file content type "{mtype}"')
    load_function = MIME_TYPES_LOADERS[mtype]

    return load_function(file_path, **kwargs)


def load_documents_txt(file_path: str, **kwargs: Any) -> list[Document]:
    """Load documents from TXT file"""
    loader = TextLoader(file_path=file_path, autodetect_encoding=True)
    return loader.load()


def load_documents_html(file_path: str, **kwargs: Any) -> list[Document]:
    """Load documents from HTML file"""
    loader = BSHTMLLoader(file_path=file_path, **kwargs)
    return loader.load()


def load_documents_csv(file_path: str, **kwargs: Any) -> list[Document]:
    """Load documents from CSV file"""
    loader = CSVLoader(
        file_path=file_path,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "doublequote": True,
            "skipinitialspace": False,
        },
    )
    return loader.load()


MIME_TYPES_LOADERS = {
    "application/pdf": load_documents_pdf,
    "text/plain": load_documents_txt,
    "text/csv": load_documents_csv,
    "text/html": load_documents_html,
}

MIME_TYPES_READERS = {
    "application/pdf": read_pdf_file,
    "text/plain": read_txt_file,
    "text/csv": read_csv_file,
    "text/html": read_html_url,
}
