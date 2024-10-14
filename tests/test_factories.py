"""Tests for the factories module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pytest

from sandbox_agent.factories import COERCE_TO, READ_ONLY, SerializerFactory


if __name__ == "__main__":
    pytest.main([__file__])


@dataclass
class TestSerializerFactory(SerializerFactory):
    """Test class for SerializerFactory."""

    normal_field: str
    read_only_field: str = field(metadata={READ_ONLY: True})
    coerce_field: int = field(metadata={COERCE_TO: str})


def test_serializer_factory_as_dict() -> None:
    """
    Test the as_dict method of SerializerFactory.

    This test covers:
    1. Normal fields are included in the output dictionary.
    2. Read-only fields are excluded from the output dictionary.
    3. Fields with COERCE_TO metadata are converted to the specified type.
    """
    test_instance = TestSerializerFactory(normal_field="normal", read_only_field="readonly", coerce_field=42)

    result = test_instance.as_dict()

    assert isinstance(result, dict)
    assert result == {"normal_field": "normal", "coerce_field": "42"}
    assert "read_only_field" not in result
    assert isinstance(result["coerce_field"], str)


@pytest.mark.parametrize(
    "field_name, field_value, expected",
    [
        ("normal_field", "test", "test"),
        ("coerce_field", 100, "100"),
    ],
)
def test_serializer_factory_field_handling(field_name: str, field_value: Any, expected: Any) -> None:
    """
    Test the handling of different field types in SerializerFactory.

    Args:
        field_name: The name of the field to test.
        field_value: The value to set for the field.
        expected: The expected value after serialization.

    This test covers:
    1. Normal fields are handled correctly.
    2. Fields with COERCE_TO metadata are converted correctly.
    """
    test_instance = TestSerializerFactory(normal_field="default", read_only_field="readonly", coerce_field=0)
    setattr(test_instance, field_name, field_value)

    result = test_instance.as_dict()

    assert result[field_name] == expected


def test_serializer_factory_read_only_field() -> None:
    """
    Test that read-only fields are always excluded from serialization.

    This test covers:
    1. Read-only fields are excluded from the output dictionary,
       regardless of their value.
    """
    test_instance = TestSerializerFactory(normal_field="normal", read_only_field="should_not_appear", coerce_field=42)

    result = test_instance.as_dict()

    assert "read_only_field" not in result


def test_serializer_factory_inheritance() -> None:
    """
    Test that SerializerFactory works correctly when inherited.

    This test covers:
    1. The as_dict method works correctly for classes that inherit from SerializerFactory.
    2. Metadata from parent classes is respected in child classes.
    """

    @dataclass
    class ChildSerializerFactory(TestSerializerFactory):
        child_field: str

    child_instance = ChildSerializerFactory(
        normal_field="parent_normal", read_only_field="parent_readonly", coerce_field=99, child_field="child_value"
    )

    result = child_instance.as_dict()

    assert isinstance(result, dict)
    assert result == {"normal_field": "parent_normal", "coerce_field": "99", "child_field": "child_value"}
    assert "read_only_field" not in result


def test_serializer_factory_empty_instance() -> None:
    """
    Test SerializerFactory with an empty instance.

    This test covers:
    1. The as_dict method works correctly for an instance with no fields.
    """

    @dataclass
    class EmptySerializerFactory(SerializerFactory):
        pass

    empty_instance = EmptySerializerFactory()

    result = empty_instance.as_dict()

    assert isinstance(result, dict)
    assert result == {}
