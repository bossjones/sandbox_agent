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
class TestDataclass(SerializerFactory):
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
        serialized = test_obj.as_dict()

        assert serialized == {
            "field1": "value1",
            "field2": 2,
            "field4": 3,
        }

    def test_as_dict_with_no_metadata(self) -> None:
        """Test that the as_dict method works correctly with no metadata."""
        @dataclasses.dataclass
        class NoMetadataClass(SerializerFactory):
            field1: str
            field2: int

        test_obj = NoMetadataClass("value1", 2)
        serialized = test_obj.as_dict()

        assert serialized == {
            "field1": "value1",
            "field2": 2,
        }

    def test_as_dict_with_custom_coerce_to(self) -> None:
        """Test that the as_dict method works correctly with a custom coerce_to function."""
        def custom_coerce(value):
            return f"coerced_{value}"

        @dataclasses.dataclass
        class CustomCoerceClass(SerializerFactory):
            field1: str
            field2: int = dataclasses.field(metadata={COERCE_TO: custom_coerce})

        test_obj = CustomCoerceClass("value1", 2)
        serialized = test_obj.as_dict()

        assert serialized == {
            "field1": "value1",
            "field2": "coerced_2",
        }
        custom_coerce_to.assert_called_once_with(3.14)
