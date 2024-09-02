"""Utility methods for unit tests."""

from __future__ import annotations

from warnings import warn


from PIL import ImageChops
from PIL.Image import Image


def images_are_identical(image_1: Image, image_2: Image):
    """
    _summary_

    Args:
    ----
        image_1 (Image): _description_
        image_2 (Image): _description_

    Raises:
    ------
        NotImplementedError: _description_

    Returns:
    -------
        _type_: _description_

    """
    if image_1.mode != image_2.mode:
        warn(
            f"Images do not have the same mode. Found {image_1.mode} and {image_2.mode}."
        )
        return False
    elif image_1.mode in ["1", "L"]:
        return ImageChops.difference(image_1, image_2).getextrema() == ((0, 0))
    elif image_1.mode == "LA":
        return ImageChops.difference(image_1, image_2).getextrema() == ((0, 0), (0, 0))
    elif image_1.mode == "RGB":
        return ImageChops.difference(image_1, image_2).getextrema() == (
            (0, 0),
            (0, 0),
            (0, 0),
        )
    elif image_1.mode == "RGBA":
        return ImageChops.difference(image_1, image_2).getextrema() == (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        )
    else:
        raise NotImplementedError(f"Unable to compare images of type {image_1.mode}.")
