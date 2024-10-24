"""collections module tests"""

from __future__ import annotations

import uuid

from typing import TYPE_CHECKING

from loguru import logger

import pytest

from sandbox_agent.ai.collections import (
    collection_exists,
    collection_name_exists,
    collections_info,
    create_collection,
    delete_collection,
    single_collection,
    update_collection,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.logging import LogCaptureFixture


@pytest.mark.cursorgenerated
def test_create_collection(caplog: LogCaptureFixture, capsys: CaptureFixture):
    """Test create_collection function"""
    result = create_collection("test_collection", {"key": "value"})
    assert result.uuid is not None
    assert result.name == "test_collection"
    assert result.cmetadata == {"key": "value"}


@pytest.mark.cursorgenerated
def test_collections_info(caplog: LogCaptureFixture, capsys: CaptureFixture):
    """Test collections_info()"""
    result = collections_info()
    assert result[0]["cmetadata"] == {"key": "value"}
    assert result[0]["name"] == "test_collection"
    assert result[0]["uuid"] is not None


@pytest.mark.cursorgenerated
def test_collection_name_exists(caplog: LogCaptureFixture, capsys: CaptureFixture):
    """Test collection_name_exists function"""
    create_collection("test_collection2", {})
    assert collection_name_exists("test_collection2") is True
    assert collection_name_exists("nonexistent_collection") is False


@pytest.mark.cursorgenerated
def test_collection_exists(caplog: LogCaptureFixture, capsys: CaptureFixture):
    """Test collection_exists function"""
    collection = create_collection("new_collection", {})
    assert collection_exists(collection.uuid) is True
    assert collection_exists(uuid.uuid4()) is False


@pytest.mark.cursorgenerated
def test_update_collection(caplog: LogCaptureFixture, capsys: CaptureFixture):
    """Test update_collection function"""
    collection = create_collection("new_collection", {})
    update_collection(collection.uuid, "updated_collection", {"new_key": "new_value"})
    updated = single_collection(collection.uuid)
    assert updated.name == "updated_collection"
    assert updated.cmetadata == {"new_key": "new_value"}


@pytest.mark.skip(reason="Not implemented")
@pytest.mark.flaky
@pytest.mark.cursorgenerated
def test_delete_collection(caplog: LogCaptureFixture, capsys: CaptureFixture):
    """Test delete_collection function"""
    collection = create_collection("new_collection", {})
    delete_collection(collection.uuid)
    assert collection_name_exists("new_collection") is False
