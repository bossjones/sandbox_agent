from __future__ import annotations


from pathlib import Path
from typing import TYPE_CHECKING

from sandbox_agent.utils import vidops

import pytest


if TYPE_CHECKING:
    pass


@pytest.fixture()
def sample_video():
    return Path("tests/fixtures/song.mp4")


def test_calculate_bitrate():
    """Test the calculate_bitrate function."""
    assert vidops.calculate_bitrate(60, 10) == 1333
    assert vidops.calculate_bitrate(120, 5) == 333
    assert vidops.calculate_bitrate(30, 20) == 5333


@pytest.mark.asyncio()
async def test_duration_video(sample_video, tmp_path, mocker):
    """Test the process_video function."""
    duration: float = await vidops.get_duration(sample_video)
    assert duration == 36.133333


@pytest.mark.asyncio()
async def test_process_video(sample_video, tmp_path, mocker):
    """Test the process_video function."""
    await vidops.process_video(sample_video)


@pytest.mark.asyncio()
async def test_process_audio(sample_video, tmp_path, mocker):
    """Test the process_audio function."""
    await vidops.process_audio(sample_video)


@pytest.mark.asyncio()
async def test_process_video_low_bitrate(sample_video, tmp_path, mocker):
    """Test the process_video function with a low bitrate scenario."""
    await vidops.process_video(sample_video)


@pytest.mark.asyncio()
async def test_process_audio_low_bitrate(sample_video, tmp_path, mocker):
    """Test the process_audio function with a low bitrate scenario."""
    await vidops.process_audio(sample_video)
