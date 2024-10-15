from __future__ import annotations

import json
import os
import pathlib

from os import PathLike
from typing import List

import aiofiles
import pandas as pd

import pytest

from sandbox_agent.utils.file_functions import (
    aio_json_loads,
    aio_read_jsonfile,
    aioread_file,
    aiowrite_file,
    check_file_size,
    expand_path_str,
    filter_audio,
    filter_gif,
    filter_images,
    filter_json,
    filter_m3u8,
    filter_media,
    filter_mkv,
    filter_pdf,
    filter_pdfs,
    filter_pth,
    filter_txts,
    filter_videos,
    filter_webm,
    fix_path,
    format_size,
    get_all_media_files_to_upload,
    get_dataframe_from_csv,
    get_files_to_upload,
    glob_file_by_extension,
    is_a_symlink,
    is_directory,
    is_file,
    print_and_append,
    rename_without_cachebuster,
    rich_display_meme_pull_list,
    rich_display_popstars_analytics,
    rich_format_followers,
    rich_likes_or_comments,
    run_aio_json_loads,
    run_tree,
    sort_dataframe,
    sort_dir_by_ctime,
    sort_dir_by_mtime,
    tilda,
    tree,
    unlink_orig_file,
)


HERE = os.path.dirname(__file__)

# from sandbox_agent.services.chroma_service import CHROMA_PATH, DATA_PATH


def test_sort_dir_by_mtime(mocker):
    mocker.patch("pathlib.Path.iterdir", return_value=[pathlib.Path("file1"), pathlib.Path("file2")])
    mocker.patch("os.path.getmtime", side_effect=[2, 1])
    result = sort_dir_by_mtime("test_dir")
    assert result == [pathlib.Path("file2"), pathlib.Path("file1")]


def test_sort_dir_by_ctime(mocker):
    mocker.patch("pathlib.Path.iterdir", return_value=[pathlib.Path("file1"), pathlib.Path("file2")])
    mocker.patch("os.path.getctime", side_effect=[2, 1])
    result = sort_dir_by_ctime("test_dir")
    assert result == [pathlib.Path("file2"), pathlib.Path("file1")]


def test_get_all_media_files_to_upload(mocker):
    mock_tree = mocker.patch("sandbox_agent.utils.file_functions.tree", return_value=["file1", "file2"])
    mock_filter_media = mocker.patch("sandbox_agent.utils.file_functions.filter_media", return_value=["file1"])
    result = get_all_media_files_to_upload("test_dir")
    assert result == ["file1"]
    mock_tree.assert_called_once_with(pathlib.Path("test_dir"))
    mock_filter_media.assert_called_once_with(["file1", "file2"])


@pytest.mark.asyncio
async def test_aio_read_jsonfile(tmp_path):
    test_data = {"key": "value"}
    json_file = tmp_path / "test.json"
    with open(json_file, "w") as f:
        json.dump(test_data, f)

    result = await aio_read_jsonfile(str(json_file))
    assert result == test_data


@pytest.mark.asyncio
async def test_aio_json_loads(tmp_path):
    test_data = {"key": "value"}
    json_file = tmp_path / "test.json"
    with open(json_file, "w") as f:
        json.dump(test_data, f)

    result = await aio_json_loads(str(json_file))
    assert result == test_data


@pytest.mark.asyncio
async def test_run_aio_json_loads(tmp_path):
    test_data = {"key": "value"}
    json_file = tmp_path / "test.json"
    with open(json_file, "w") as f:
        json.dump(test_data, f)

    result = await run_aio_json_loads(str(json_file))
    assert result == test_data


# def test_filter_pdfs():
#     d = tree(DATA_PATH)
#     result = filter_pdfs(d)
#     expected = [
#         "opencv-tutorial-readthedocs-io-en-latest.pdf",
#         "pillow-readthedocs-io-en-latest.pdf",
#         "rich-readthedocs-io-en-latest.pdf",
#         "Understanding_Climate_Change.pdf",
#     ]

#     for i in result:
#         assert i.name in expected


def test_filter_media(mocker):
    mock_filter_images = mocker.patch("sandbox_agent.utils.file_functions.filter_images", return_value=["file1.png"])
    mock_filter_videos = mocker.patch("sandbox_agent.utils.file_functions.filter_videos", return_value=["file2.mp4"])
    result = filter_media(["file1.png", "file2.mp4", "file3.txt"])
    assert result == ["file1.png", "file2.mp4"]
    mock_filter_images.assert_called_once_with(["file1.png", "file2.mp4", "file3.txt"])
    mock_filter_videos.assert_called_once_with(["file1.png", "file2.mp4", "file3.txt"])


def test_get_dataframe_from_csv(mocker):
    mock_read_csv = mocker.patch("pandas.read_csv", return_value=pd.DataFrame({"col1": [1], "col2": [2]}))
    result = get_dataframe_from_csv("/Users/malcolm/dev/bossjones/sandbox_agent/test.csv")
    assert result.equals(
        pd.DataFrame({"col1": [1], "col2": [2]})
    ), f"Expected DataFrame: {pd.DataFrame({'col1': [1], 'col2': [2]})}, but got: {result}"
    mock_read_csv.assert_called_once_with("/Users/malcolm/dev/bossjones/sandbox_agent/test.csv")


def test_rich_format_followers():
    result = rich_format_followers(1000000)
    assert result == "[bold bright_white]1000000[/bold bright_white]"


@pytest.mark.asyncio
async def test_aiowrite_file(tmp_path):
    test_data = "Test content"
    test_file = tmp_path / "test.txt"
    await aiowrite_file(test_data, str(tmp_path), "test", "txt")
    assert test_file.read_text() == test_data


@pytest.mark.asyncio
async def test_aioread_file(tmp_path):
    test_data = "Test content"
    test_file = tmp_path / "test.txt"
    test_file.write_text(test_data)

    result = await aioread_file(str(tmp_path), "test", "txt")
    assert result == test_data


def test_rich_likes_or_comments():
    result = rich_likes_or_comments(10000)
    assert result == "[bold bright_yellow]10000[/bold bright_yellow]"


def test_rich_display_meme_pull_list(mocker):
    mock_console = mocker.patch("sandbox_agent.utils.file_functions.Console.print")
    df = pd.DataFrame(
        {
            "Account": ["acc1"],
            "Social": ["soc1"],
            "Total Followers": [1000],
            "Total Likes": [100],
            "Total Comments": [10],
            "Total Posts": [5],
            "Start Date": ["2021-01-01"],
            "End Date": ["2021-01-02"],
            "ERDay": [0.1],
            "ERpost": [0.2],
            "Average Likes": [20],
            "Average Comments": [2],
            "Links": ["link1"],
        }
    )
    rich_display_meme_pull_list(df)
    mock_console.assert_called_once()


def test_rich_display_popstars_analytics(mocker):
    mock_console = mocker.patch("sandbox_agent.utils.file_functions.Console.print")
    df = pd.DataFrame(
        {
            "Social": ["soc1"],
            "Author": ["auth1"],
            "Url": ["url1"],
            "Likes": [100],
            "Comments": [10],
            "ER": [0.1],
            "Text": ["text1"],
            "Date": ["2021-01-01"],
            "Media 1": ["media1"],
        }
    )
    rich_display_popstars_analytics(df)
    mock_console.assert_called_once()


def test_glob_file_by_extension(mocker):
    mock_glob = mocker.patch("glob.glob", return_value=["file1.mp4"])
    result = glob_file_by_extension("test_dir", "*.mp4", recursive=False)
    assert result == ["file1.mp4"]
    mock_glob.assert_called_once_with("test_dir/*.mp4", recursive=False)


def test_unlink_orig_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    assert test_file.exists()

    unlink_orig_file(str(test_file))
    assert not test_file.exists()


def test_get_files_to_upload(tmp_path):
    image_file = tmp_path / "image.jpg"
    video_file = tmp_path / "video.mp4"
    text_file = tmp_path / "text.txt"

    image_file.touch()
    video_file.touch()
    text_file.touch()

    result = get_files_to_upload(str(tmp_path))
    assert len(result) == 2
    assert str(image_file) in result
    assert str(video_file) in result
    assert str(text_file) not in result


def test_run_tree(tmp_path):
    image_file = tmp_path / "image.jpg"
    video_file = tmp_path / "video.mp4"
    text_file = tmp_path / "text.txt"

    image_file.touch()
    video_file.touch()
    text_file.touch()

    result = run_tree(str(tmp_path))
    assert len(result) == 2
    assert str(image_file) in result
    assert str(video_file) in result
    assert str(text_file) not in result


def test_print_and_append(mocker):
    mock_print = mocker.patch("builtins.print")
    dir_listing = []
    print_and_append(dir_listing, "test_str", silent=False)
    assert dir_listing == ["test_str"]
    mock_print.assert_called_once_with("test_str")


def test_check_file_size(mocker):
    mock_stat = mocker.patch("pathlib.Path.stat", return_value=mocker.Mock(st_size=1024))
    result = check_file_size("/Users/malcolm/dev/bossjones/sandbox_agent/test_file")
    assert result == (False, "Is file size greater than 50000000: False")


def test_is_file(mocker):
    mock_is_file = mocker.patch("pathlib.Path.is_file", return_value=True)
    result = is_file("test_file")
    assert result is True
    mock_is_file.assert_called_once_with()


def test_is_directory(mocker):
    mock_is_dir = mocker.patch("pathlib.Path.is_dir", return_value=True)
    result = is_directory("test_dir")
    assert result is True
    mock_is_dir.assert_called_once_with()


def test_is_a_symlink(mocker):
    mock_is_symlink = mocker.patch("pathlib.Path.is_symlink", return_value=True)
    result = is_a_symlink("test_symlink")
    assert result is True
    mock_is_symlink.assert_called_once_with()


def test_expand_path_str(mocker):
    mock_expanduser = mocker.patch("pathlib.Path.expanduser", return_value=pathlib.Path("/home/user"))
    result = expand_path_str("~")
    assert result == pathlib.Path("/home/user")
    mock_expanduser.assert_called_once_with()


def test_tilda(mocker):
    mock_expanduser = mocker.patch("pathlib.Path.expanduser", return_value=pathlib.Path("/home/user"))
    result = tilda("~")
    assert result == "/home/user"
    mock_expanduser.assert_called_once_with()


def test_fix_path(mocker):
    mock_is_file = mocker.patch("sandbox_agent.utils.file_functions.is_file", return_value=True)
    mock_tilda = mocker.patch("sandbox_agent.utils.file_functions.tilda", return_value="/home/user/file")
    result = fix_path("~")
    assert result == "/home/user/file"
    mock_is_file.assert_called_once_with("/home/user/file")
    mock_tilda.assert_called_once_with("~")


def test_filter_pth(tmp_path):
    pth_file = tmp_path / "model.pth"
    txt_file = tmp_path / "text.txt"
    pth_file.touch()
    txt_file.touch()

    result = filter_pth([str(pth_file), str(txt_file)])
    assert result == [str(pth_file)]


def test_filter_json(tmp_path):
    json_file = tmp_path / "data.json"
    txt_file = tmp_path / "text.txt"
    json_file.touch()
    txt_file.touch()

    result = filter_json([str(json_file), str(txt_file)])
    assert result == [str(json_file)]


def test_rename_without_cachebuster(tmp_path):
    file_with_cb = tmp_path / "image.jpg?updatedAt=123456"
    file_with_cb.touch()

    result = rename_without_cachebuster([str(file_with_cb)])
    assert result == [str(tmp_path / "image.jpg")]
    assert (tmp_path / "image.jpg").exists()


def test_filter_videos(tmp_path):
    video_file = tmp_path / "video.mp4"
    txt_file = tmp_path / "text.txt"
    video_file.touch()
    txt_file.touch()

    result = filter_videos([str(video_file), str(txt_file)])
    assert result == [str(video_file)]


def test_filter_audio(tmp_path):
    audio_file = tmp_path / "audio.mp3"
    txt_file = tmp_path / "text.txt"
    audio_file.touch()
    txt_file.touch()

    result = filter_audio([str(audio_file), str(txt_file)])
    assert result == [str(audio_file)]


def test_filter_gif(tmp_path):
    gif_file = tmp_path / "image.gif"
    txt_file = tmp_path / "text.txt"
    gif_file.touch()
    txt_file.touch()

    result = filter_gif([str(gif_file), str(txt_file)])
    assert result == [str(gif_file)]


def test_filter_mkv(tmp_path):
    mkv_file = tmp_path / "video.mkv"
    txt_file = tmp_path / "text.txt"
    mkv_file.touch()
    txt_file.touch()

    result = filter_mkv([str(mkv_file), str(txt_file)])
    assert result == [str(mkv_file)]


def test_filter_m3u8(tmp_path):
    m3u8_file = tmp_path / "playlist.m3u8"
    txt_file = tmp_path / "text.txt"
    m3u8_file.touch()
    txt_file.touch()

    result = filter_m3u8([str(m3u8_file), str(txt_file)])
    assert result == [str(m3u8_file)]


def test_filter_webm(tmp_path):
    webm_file = tmp_path / "video.webm"
    txt_file = tmp_path / "text.txt"
    webm_file.touch()
    txt_file.touch()

    result = filter_webm([str(webm_file), str(txt_file)])
    assert result == [str(webm_file)]


def test_filter_images(tmp_path):
    image_file = tmp_path / "image.jpg"
    txt_file = tmp_path / "text.txt"
    image_file.touch()
    txt_file.touch()

    result = filter_images([str(image_file), str(txt_file)])
    assert result == [str(image_file)]


def test_filter_pdfs(tmp_path):
    pdf_file = tmp_path / "document.pdf"
    txt_file = tmp_path / "text.txt"
    pdf_file.touch()
    txt_file.touch()

    result = filter_pdfs([str(pdf_file), str(txt_file)])
    assert result == [str(pdf_file)]


def test_filter_txts(tmp_path):
    txt_file1 = tmp_path / "text1.txt"
    txt_file2 = tmp_path / "text2.txt"
    pdf_file = tmp_path / "document.pdf"
    txt_file1.touch()
    txt_file2.touch()
    pdf_file.touch()

    result = filter_txts([str(txt_file1), str(txt_file2), str(pdf_file)])
    assert set(result) == {str(txt_file1), str(txt_file2)}


def test_filter_pdf(tmp_path):
    pdf_file = tmp_path / "document.pdf"
    txt_file = tmp_path / "text.txt"
    pdf_file.touch()
    txt_file.touch()

    result = filter_pdf([str(pdf_file), str(txt_file)])
    assert result == [str(pdf_file)]


def test_sort_dataframe():
    df = pd.DataFrame({"A": [3, 1, 2], "B": ["c", "a", "b"]})
    sorted_df = sort_dataframe(df, columns=["A"], ascending=(True,))
    assert sorted_df["A"].tolist() == [1, 2, 3]
    assert sorted_df["B"].tolist() == ["a", "b", "c"]


def test_format_size():
    assert format_size(1024) == "1.00 KB"
    assert format_size(1024 * 1024) == "1.00 MB"
    assert format_size(1024 * 1024 * 1024) == "1.00 GB"
    assert format_size(500) == "500 B"
    assert format_size(1500) == "1.46 KB"
    assert format_size(1500000) == "1.43 MB"


def test_tree(tmp_path):
    (tmp_path / "file1.txt").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.txt").touch()

    result = tree(tmp_path)
    assert len(result) == 3
    assert any(str(tmp_path) in str(path) for path in result)
    assert any("file1.txt" in str(path) for path in result)
    assert any("dir1" in str(path) for path in result)
    assert any("file2.txt" in str(path) for path in result)
