# """Index module tests"""

# from __future__ import annotations

# from pathlib import Path
# from typing import TYPE_CHECKING

# from h11 import Response
# from langchain.docstore.document import Document
# from langchain_text_splitters import NLTKTextSplitter
# from langchain_text_splitters.character import RecursiveCharacterTextSplitter
# from loguru import logger
# from requests import HTTPError

# import pytest

# from sandbox_agent.ai.collections import create_collection
# from sandbox_agent.ai.index import (
#     add_document,
#     create_splitter,
#     document_has_changed,
#     documents_metadata,
#     load_pdf_file,
#     select_load_link_options,
#     split_document,
#     update_links_documents,
# )
# from sandbox_agent.aio_settings import aiosettings
# from sandbox_agent.vendored.brevia.brevia.settings import get_settings
# from vcr.request import Request as VCRRequest


# if TYPE_CHECKING:
#     from _pytest.capture import CaptureFixture
#     from _pytest.logging import LogCaptureFixture


# def test_load_pdf_file(caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
#     """Test the load_pdf_file method.

#     This test verifies that a PDF file is loaded correctly into a collection.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#     """
#     collection = create_collection("test_collection", {})
#     file_path = f"{Path(__file__).parent}/files/docs/empty.pdf"
#     result = load_pdf_file(file_path=file_path, collection_name=collection.name)
#     assert result == 1


# def test_load_pdf_file_fail(caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
#     """Test the load_pdf_file method for failure.

#     This test checks that a FileNotFoundError is raised when a non-existent file is loaded.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#     """
#     file_path = f"{Path(__file__).parent}/files/notfound.pdf"
#     with pytest.raises(FileNotFoundError) as exc:
#         load_pdf_file(file_path=file_path, collection_name="test")
#     assert str(exc.value) == file_path


# def test_update_links_documents(caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
#     """Test the update_links_documents method.

#     This test verifies that the update_links_documents method returns zero results
#     when no documents are updated.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#     """
#     result = update_links_documents("test")
#     assert result == 0

#     create_collection("test", {})
#     doc1 = Document(page_content="some", metadata={"type": "links"})
#     add_document(document=doc1, collection_name="test")
#     result = update_links_documents("test")
#     assert result == 0


# @pytest.mark.flaky
# @pytest.mark.skip(reason="Not implemented")
# def test_document_has_changed(caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
#     """Test the document_has_changed method.

#     This test checks if the document_has_changed method correctly identifies changes
#     in document metadata.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#     """
#     collection = create_collection("test", {})

#     doc1 = Document(page_content="some", metadata={})
#     add_document(document=doc1, collection_name="test", document_id="1")
#     result = document_has_changed(doc1, collection.uuid, "1")
#     assert result is False

#     doc1.metadata = {"type": "links"}
#     result = document_has_changed(doc1, collection.uuid, "1")
#     assert result is True

#     add_document(document=Document(page_content="another"), collection_name="test", document_id="1")
#     result = document_has_changed(doc1, collection.uuid, "1")
#     assert result is True


# def _raise_http_err():
#     raise HTTPError("404 Client Error", response=Response(headers=[], status_code=404))


# # @patch("brevia.index.load_file.requests.get")
# # def test_update_links_documents_http_error(mock_get):
# #     """Test update_links_documents method with HTTP error"""
# #     collection = create_collection("test", {})
# #     doc1 = Document(page_content="some", metadata={"type": "links", "url": "http://example.com"})
# #     add_document(document=doc1, collection_name="test", document_id="1")

# #     mock_get.return_value.status_code = 404
# #     mock_get.return_value.text = "404 Client Error"
# #     mock_get.return_value.raise_for_status = _raise_http_err

# #     result = update_links_documents("test")
# #     assert result == 0
# #     meta = documents_metadata(collection_id=collection.uuid, document_id="1")
# #     assert meta[0]["cmetadata"]["http_error"] == "404"

# #     mock_get.return_value.status_code = 200
# #     mock_get.return_value.text = "changed"

# #     def donothing():
# #         pass

# #     mock_get.return_value.raise_for_status = donothing

# #     result = update_links_documents("test")
# #     assert result == 1
# #     meta = documents_metadata(collection_id=collection.uuid, document_id="1")
# #     assert meta[0]["cmetadata"].get("http_error") is None


# def test_select_load_link_options(caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
#     """Test the select_load_link_options method.

#     This test verifies that the correct selector is returned based on the URL.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#     """
#     options = [{"url": "example.com", "selector": "test"}]
#     result = select_load_link_options(url="example.com", options=options)
#     assert result == {"selector": "test"}

#     result = select_load_link_options(url="example.com/somepath", options=options)
#     assert result == {"selector": "test"}

#     result = select_load_link_options(url="someurl.org", options=options)
#     assert result == {}


# @pytest.mark.vcronly
# @pytest.mark.default_cassette("test_custom_split.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_custom_split(caplog: LogCaptureFixture, capsys: CaptureFixture, vcr: VCRRequest) -> None:
#     """Test the split_document method with a custom splitter class.

#     This test checks that the document is split correctly using a custom splitter.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#         vcr (VCRRequest): The VCR request object for recording/replaying HTTP interactions.
#     """
#     doc1 = Document(page_content="some content? no", metadata={"type": "questions"})
#     cls = "langchain_text_splitters.character.RecursiveCharacterTextSplitter"
#     texts = split_document(doc1, {"splitter": cls})

#     assert len(texts) == 1
#     assert vcr.play_count > 0 # type: ignore


# @pytest.mark.vcronly
# @pytest.mark.default_cassette("test_add_document_custom_split.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_add_document_custom_split(caplog: LogCaptureFixture, capsys: CaptureFixture, vcr: VCRRequest) -> None:
#     """Test the add_document method with a custom splitter in settings.

#     This test verifies that a document is added correctly using a custom splitter.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#         vcr (VCRRequest): The VCR request object for recording/replaying HTTP interactions.
#     """
#     settings = aiosettings
#     current_splitter = settings.text_splitter
#     settings.text_splitter = {"splitter": "langchain_text_splitters.character.RecursiveCharacterTextSplitter"}
#     doc1 = Document(page_content="some content? no", metadata={"type": "questions"})
#     num = add_document(document=doc1, collection_name="test")
#     assert num == 1
#     assert vcr.play_count > 0 # type: ignore

#     settings.text_splitter = current_splitter


# @pytest.mark.vcronly
# @pytest.mark.default_cassette("test_create_splitter_chunk_params.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_create_splitter_chunk_params(caplog: LogCaptureFixture, capsys: CaptureFixture, vcr: VCRRequest) -> None:
#     """Test the create_splitter method.

#     This test verifies that the splitter is created with the correct chunk parameters.

#     Args:
#         caplog (LogCaptureFixture): Fixture to capture log messages.
#         capsys (CaptureFixture): Fixture to capture stdout and stderr.
#         vcr (VCRRequest): The VCR request object for recording/replaying HTTP interactions.
#     """
#     splitter = create_splitter({"chunk_size": 2222, "chunk_overlap": 333})

#     assert isinstance(splitter, NLTKTextSplitter)
#     assert splitter._chunk_size == 2222
#     assert splitter._chunk_overlap == 333
#     assert vcr.play_count > 0 # type: ignore

#     splitter = create_splitter({})

#     assert isinstance(splitter, NLTKTextSplitter)
#     assert splitter._chunk_size == get_settings().text_chunk_size
#     assert splitter._chunk_overlap == get_settings().text_chunk_overlap
#     assert vcr.play_count > 0 # type: ignore

#     custom_splitter = {
#         "splitter": "langchain_text_splitters.character.RecursiveCharacterTextSplitter",
#         "chunk_size": 1111,
#         "chunk_overlap": 555,
#     }
#     splitter = create_splitter(
#         {
#             "chunk_size": 3333,
#             "chunk_overlap": 444,
#             "text_splitter": custom_splitter,
#         }
#     )

#     assert isinstance(splitter, RecursiveCharacterTextSplitter)
#     assert splitter._chunk_size == 1111
#     assert vcr.play_count > 0 # type: ignore
