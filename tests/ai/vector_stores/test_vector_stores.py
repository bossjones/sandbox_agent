# tests/test_vector_stores.py

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING, Any

from langchain.docstore.document import Document
from langchain_community.vectorstores import SKLearnVectorStore

import pytest

from sandbox_agent.ai.vector_stores import SKLearnVectorStoreAdapter, VectorStoreFactory
from sandbox_agent.aio_settings import aiosettings


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from vcr.request import Request as VCRRequest

    from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def set_aiosettings(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SKLEARN_PERSIST_PATH", os.getenv("SKLEARN_PERSIST_PATH", "./test.db"))


@pytest.fixture()
def mock_documents() -> list[Document]:
    """
    Fixture to create a list of mock documents for testing.

    Returns:
    -------
        list[Document]: A list of mock Document objects.
    """
    return [
        Document(page_content="This is a test document", metadata={"source": "test"}),
        Document(page_content="Another test document", metadata={"source": "test2"}),
    ]


@pytest.mark.integration()
@pytest.mark.vcronly()
@pytest.mark.default_cassette("test_sklearn_vector_store_adapter.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
def test_sklearn_vector_store_adapter_initialization(
    mock_documents: list[Document],
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    vcr: VCRRequest,
    set_aiosettings: FixtureRequest,
) -> None:
    """
    Test the initialization of SKLearnVectorStoreAdapter.

    Args:
    ----
        mock_documents (list[Document]): List of mock documents.
        tmp_path (pathlib.Path): Temporary path for testing.
        mocker (MockerFixture): Pytest mocker fixture.
        vcr (VCRRequest): VCR request object for recording/replaying HTTP interactions.
    """
    # mocker.patch("sandbox_agent.aio_settings.aiosettings.sklearn_persist_path", str(tmp_path / "test_vector_store"))

    adapter = SKLearnVectorStoreAdapter(documents=mock_documents)

    assert isinstance(adapter.vectorstore, SKLearnVectorStore)
    assert os.path.exists(aiosettings.sklearn_persist_path)
    assert vcr.play_count > 0
    # os.path.remove(aiosettings.sklearn_persist_path)


# @pytest.mark.integration
# @pytest.mark.vcronly()
# @pytest.mark.default_cassette("test_sklearn_vector_store_adapter_add_documents.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_sklearn_vector_store_adapter_add_documents(
#     mock_documents: list[Document],
#     tmp_path: pathlib.Path,
#     mocker: MockerFixture,
#     vcr: VCRRequest,
# ) -> None:
#     """
#     Test adding documents to SKLearnVectorStoreAdapter.

#     Args:
#     ----
#         mock_documents (list[Document]): List of mock documents.
#         tmp_path (pathlib.Path): Temporary path for testing.
#         mocker (MockerFixture): Pytest mocker fixture.
#         vcr (VCRRequest): VCR request object for recording/replaying HTTP interactions.
#     """
#     mocker.patch("sandbox_agent.aio_settings.aiosettings.sklearn_persist_path", str(tmp_path / "test_vector_store"))

#     adapter = SKLearnVectorStoreAdapter()
#     adapter.add_documents(mock_documents)

#     assert len(adapter.vectorstore.docstore._dict) == len(mock_documents)
#     assert vcr.play_count > 0

# @pytest.mark.integration
# @pytest.mark.vcronly()
# @pytest.mark.default_cassette("test_sklearn_vector_store_adapter_search.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_sklearn_vector_store_adapter_search(
#     mock_documents: list[Document],
#     tmp_path: pathlib.Path,
#     mocker: MockerFixture,
#     vcr: VCRRequest,
# ) -> None:
#     """
#     Test searching documents in SKLearnVectorStoreAdapter.

#     Args:
#     ----
#         mock_documents (list[Document]): List of mock documents.
#         tmp_path (pathlib.Path): Temporary path for testing.
#         mocker (MockerFixture): Pytest mocker fixture.
#         vcr (VCRRequest): VCR request object for recording/replaying HTTP interactions.
#     """
#     mocker.patch("sandbox_agent.aio_settings.aiosettings.sklearn_persist_path", str(tmp_path / "test_vector_store"))

#     adapter = SKLearnVectorStoreAdapter(documents=mock_documents)
#     results = adapter.search("test document", k=1)

#     assert len(results) == 1
#     assert "test document" in results[0].page_content
#     assert vcr.play_count > 0

# @pytest.mark.integration
# @pytest.mark.vcronly()
# @pytest.mark.default_cassette("test_vector_store_factory.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_vector_store_factory(vcr: VCRRequest) -> None:
#     """
#     Test the VectorStoreFactory.

#     Args:
#     ----
#         vcr (VCRRequest): VCR request object for recording/replaying HTTP interactions.
#     """
#     sklearn_store = VectorStoreFactory.create("sklearn")
#     assert isinstance(sklearn_store, type(SKLearnVectorStore))

#     with pytest.raises(ValueError):
#         VectorStoreFactory.create("unsupported_store")

#     assert vcr.play_count > 0

# @pytest.mark.integration
# @pytest.mark.vcronly()
# @pytest.mark.default_cassette("test_sklearn_vector_store_adapter_persist_load.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_sklearn_vector_store_adapter_persist_load(
#     mock_documents: list[Document],
#     tmp_path: pathlib.Path,
#     mocker: MockerFixture,
#     vcr: VCRRequest,
# ) -> None:
#     """
#     Test persisting and loading SKLearnVectorStoreAdapter.

#     Args:
#     ----
#         mock_documents (list[Document]): List of mock documents.
#         tmp_path (pathlib.Path): Temporary path for testing.
#         mocker (MockerFixture): Pytest mocker fixture.
#         vcr (VCRRequest): VCR request object for recording/replaying HTTP interactions.
#     """
#     persist_path = str(tmp_path / "test_vector_store")
#     mocker.patch("sandbox_agent.aio_settings.aiosettings.sklearn_persist_path", persist_path)

#     # Create and persist adapter
#     adapter1 = SKLearnVectorStoreAdapter(documents=mock_documents)
#     adapter1.persist()

#     # Create new adapter and load
#     adapter2 = SKLearnVectorStoreAdapter()
#     adapter2.load()

#     assert len(adapter2.vectorstore.docstore._dict) == len(mock_documents)
#     assert vcr.play_count > 0

# @pytest.mark.integration
# @pytest.mark.vcronly()
# @pytest.mark.default_cassette("test_sklearn_vector_store_adapter_get_vectorstore.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_sklearn_vector_store_adapter_get_vectorstore(
#     mock_documents: list[Document],
#     tmp_path: pathlib.Path,
#     mocker: MockerFixture,
#     vcr: VCRRequest,
# ) -> None:
#     """
#     Test getting the vectorstore from SKLearnVectorStoreAdapter.

#     Args:
#     ----
#         mock_documents (list[Document]): List of mock documents.
#         tmp_path (pathlib.Path): Temporary path for testing.
#         mocker (MockerFixture): Pytest mocker fixture.
#         vcr (VCRRequest): VCR request object for recording/replaying HTTP interactions.
#     """
#     mocker.patch("sandbox_agent.aio_settings.aiosettings.sklearn_persist_path", str(tmp_path / "test_vector_store"))

#     adapter = SKLearnVectorStoreAdapter(documents=mock_documents)
#     vectorstore = adapter.get_vectorstore()

#     assert isinstance(vectorstore, SKLearnVectorStore)
#     assert vcr.play_count > 0
