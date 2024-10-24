"""collections_io module tests"""

from __future__ import annotations

from os import unlink
from pathlib import Path

import pytest

from sandbox_agent.ai.collections import create_collection
from sandbox_agent.utils.collections_io import export_collection_data, import_collection_data


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
def test_export_collection_data():
    """Test export_collection_data function"""
    folder_path = f"{Path(__file__).parent.parent}/files"
    with pytest.raises(ValueError) as exc:
        export_collection_data(folder_path=folder_path, collection="test")
    assert str(exc.value) == "Collection 'test' was not found"


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
def test_export_collection_data_errors():
    """Test export_collection_data function"""
    create_collection("empty", {})
    folder_path = f"{Path(__file__).parent.parent}/files"
    csv_path = f"{folder_path}/empty-collection.csv"
    Path(csv_path).touch()
    with pytest.raises(ValueError) as exc:
        export_collection_data(folder_path=folder_path, collection="empty")
    assert str(exc.value) == f"CSV file {csv_path} already exists, exiting"
    unlink(csv_path)
    csv_path = f"{folder_path}/empty-embedding.csv"
    Path(csv_path).touch()
    with pytest.raises(ValueError) as exc:
        export_collection_data(folder_path=folder_path, collection="empty")
    assert str(exc.value) == f"CSV file {csv_path} already exists, exiting"
    unlink(csv_path)


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
def test_import_collection_data():
    """Test import_collection_data function"""
    create_collection("test", {})
    folder_path = f"{Path(__file__).parent.parent}/files"
    with pytest.raises(ValueError) as exc:
        import_collection_data(folder_path=folder_path, collection="test")
    assert str(exc.value) == "Collection 'test' already exists, exiting"


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
def test_import_collection_data_errors():
    """Test import_collection_data function"""
    folder_path = f"{Path(__file__).parent.parent}/files"
    csv_path = f"{folder_path}/empty-collection.csv"
    with pytest.raises(ValueError) as exc:
        import_collection_data(folder_path=folder_path, collection="empty")
    assert str(exc.value) == f"CSV file {csv_path} not found, exiting"

    Path(csv_path).touch()
    csv_em_path = f"{folder_path}/empty-embedding.csv"
    with pytest.raises(ValueError) as exc:
        import_collection_data(folder_path=folder_path, collection="empty")
    assert str(exc.value) == f"CSV file {csv_em_path} not found, exiting"
    unlink(csv_path)
