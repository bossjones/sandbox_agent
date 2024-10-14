"""Tests for the SerializerFactory class."""

from __future__ import annotations

import dataclasses

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from sandbox_agent.factories import COERCE_TO, READ_ONLY, SerializerFactory


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@dataclasses.dataclass
class TestDataclass:
    """A test dataclass for serialization."""

    field1: str
    field2: int
    field3: datetime = dataclasses.field(metadata={READ_ONLY: True})
    field4: float = dataclasses.field(metadata={COERCE_TO: int})


class TestSerializerFactory:
    """Tests for the SerializerFactory class."""

    def test_as_dict(self) -> None:
        """Test that the as_dict method serializes the dataclass correctly."""
        test_obj = TestDataclass("value1", 2, datetime(2023, 5, 1, 12, 0, 0), 3.14)
        serializer = SerializerFactory()
        serialized = serializer.as_dict()

        assert serialized == {
            "field1": "value1",
            "field2": 2,
            "field4": 3,
        }

    def test_as_dict_with_no_metadata(self, mocker: MockerFixture) -> None:
        """Test that the as_dict method works correctly with no metadata."""
        test_obj = TestDataclass("value1", 2, datetime(2023, 5, 1, 12, 0, 0), 3.14)
        mocker.patch.object(dataclasses, "fields", return_value=[mocker.Mock(metadata={})])
        serializer = SerializerFactory()
        serialized = serializer.as_dict()

        assert serialized == dataclasses.asdict(serializer)

    def test_as_dict_with_custom_coerce_to(self, mocker: MockerFixture) -> None:
        """Test that the as_dict method works correctly with a custom coerce_to function."""
        test_obj = TestDataclass("value1", 2, datetime(2023, 5, 1, 12, 0, 0), 3.14)
        custom_coerce_to = mocker.Mock(return_value="coerced_value")
        mocker.patch.object(
            dataclasses,
            "fields",
            return_value=[mocker.Mock(metadata={COERCE_TO: custom_coerce_to}, name="field4")],
        )
        serializer = SerializerFactory()
        serialized = serializer.as_dict()

        assert serialized == {
            "field1": "value1",
            "field2": 2,
            "field4": "coerced_value",
        }
        custom_coerce_to.assert_called_once_with(3.14)
