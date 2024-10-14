from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import pytest

from pytest_mock import MockerFixture

from src.sandbox_agent.utils import (
    AsyncFilter,
    assert_never,
    async_run,
    deduplicate_iterables,
    get_or_create_event_loop,
    is_coroutine,
    is_dict,
    make_optional,
    module_exists,
    temp_env_update,
)


def test_async_run(mocker: MockerFixture) -> None:
    """Test that async_run executes a coroutine and returns the result."""

    async def coro() -> int:
        return 42

    # mocker.patch("src.sandbox_agent.utils.get_or_create_event_loop")
    result = async_run(coro())
    assert result == 42


def test_is_coroutine(mocker: MockerFixture) -> None:
    """Test that is_coroutine correctly identifies coroutine functions."""

    async def coro() -> None:
        pass

    def func() -> None:
        pass

    assert is_coroutine(coro)
    assert not is_coroutine(func)


def test_module_exists(mocker: MockerFixture) -> None:
    """Test that module_exists correctly checks for module existence."""
    mocker.patch("src.sandbox_agent.utils.find_spec", return_value=None)
    assert not module_exists("nonexistent_module")

    mocker.patch("src.sandbox_agent.utils.find_spec", return_value=mocker.Mock())
    assert module_exists("existent_module")


# def test_temp_env_update(ctx_monkeypatch: MonkeyPatch) -> None:
#     """Test that temp_env_update temporarily updates environment variables."""
#     monkeypatch.setenv("EXISTING_VAR", "original_value")

#     with temp_env_update({"NEW_VAR": "new_value", "EXISTING_VAR": "updated_value"}):
#         assert monkeypatch.getenv("NEW_VAR") == "new_value"
#         assert monkeypatch.getenv("EXISTING_VAR") == "updated_value"

#     assert monkeypatch.getenv("NEW_VAR") is None
#     assert monkeypatch.getenv("EXISTING_VAR") == "original_value"


# def test_monkeypatch_context_manager(mocker: MockerFixture, ctx_monkeypatch: ctx_MonkeyPatch) -> None:
#     """Test that the monkeypatch context manager temporarily replaces a method."""

#     class TestClass:
#         def method(self, arg: int) -> int:
#             return arg

#     def patched_method(_original_method: Any, arg: int) -> int:
#         return arg * 2

#     test_instance = TestClass()

#     with monkeypatch(test_instance, "method", patched_method):
#         assert test_instance.method(21) == 42

#     assert test_instance.method(21) == 21


def test_get_or_create_event_loop(mocker: MockerFixture) -> None:
    """Test that get_or_create_event_loop returns an event loop."""
    mocker.patch("src.sandbox_agent.utils.asyncio.get_event_loop", side_effect=RuntimeError)
    mocker.patch("src.sandbox_agent.utils.asyncio.new_event_loop")
    mocker.patch("src.sandbox_agent.utils.asyncio.set_event_loop")

    loop = get_or_create_event_loop()
    assert loop is not None


def test_assert_never(capsys: CaptureFixture[str]) -> None:
    """Test that assert_never raises an AssertionError."""
    with pytest.raises(AssertionError):
        assert_never(42)

    # captured = capsys.readouterr()
    # assert "Unhandled type: int" in captured.out


def test_make_optional() -> None:
    """Test that make_optional returns the input value."""
    assert make_optional(42) == 42
    assert make_optional(None) is None


def test_is_dict() -> None:
    """Test that is_dict correctly identifies dictionaries."""
    assert is_dict({})
    assert is_dict({"a": 1})
    assert not is_dict([])
    assert not is_dict(42)


def test_deduplicate_iterables() -> None:
    """Test that deduplicate_iterables returns unique items in order."""
    assert deduplicate_iterables([1, 2, 3], [2, 3, 4]) == [1, 2, 3, 4]
    assert deduplicate_iterables([1, 2, 3], [3, 2, 1]) == [1, 2, 3]
    assert deduplicate_iterables([], []) == []


# @pytest.mark.asyncio
# async def test_async_filter(mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
#     """Test that AsyncFilter filters items and logs progress."""

#     async def is_even(x: int) -> bool:
#         return x % 2 == 0

#     items = [1, 2, 3, 4, 5]
#     filtered = [item async for item in AsyncFilter(is_even, items)]
#     assert filtered == [2, 4]

#     assert "Filtered 2 out of 5 items" in caplog.text
