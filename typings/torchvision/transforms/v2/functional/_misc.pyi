"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple
from torchvision import tv_tensors
from ._utils import _register_kernel_internal

def normalize(inpt: torch.Tensor, mean: List[float], std: List[float], inplace: bool = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.Normalize` for details."""
    ...

@_register_kernel_internal(normalize, torch.Tensor)
@_register_kernel_internal(normalize, tv_tensors.Image)
def normalize_image(image: torch.Tensor, mean: List[float], std: List[float], inplace: bool = ...) -> torch.Tensor:
    ...

@_register_kernel_internal(normalize, tv_tensors.Video)
def normalize_video(video: torch.Tensor, mean: List[float], std: List[float], inplace: bool = ...) -> torch.Tensor:
    ...

def gaussian_blur(inpt: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = ...) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.GaussianBlur` for details."""
    ...

@_register_kernel_internal(gaussian_blur, torch.Tensor)
@_register_kernel_internal(gaussian_blur, tv_tensors.Image)
def gaussian_blur_image(image: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = ...) -> torch.Tensor:
    ...

@_register_kernel_internal(gaussian_blur, tv_tensors.Video)
def gaussian_blur_video(video: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = ...) -> torch.Tensor:
    ...

def to_dtype(inpt: torch.Tensor, dtype: torch.dtype = ..., scale: bool = ...) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.ToDtype` for details."""
    ...

@_register_kernel_internal(to_dtype, torch.Tensor)
@_register_kernel_internal(to_dtype, tv_tensors.Image)
def to_dtype_image(image: torch.Tensor, dtype: torch.dtype = ..., scale: bool = ...) -> torch.Tensor:
    ...

def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = ...) -> torch.Tensor:
    """[DEPRECATED] Use to_dtype() instead."""
    ...

@_register_kernel_internal(to_dtype, tv_tensors.Video)
def to_dtype_video(video: torch.Tensor, dtype: torch.dtype = ..., scale: bool = ...) -> torch.Tensor:
    ...

def sanitize_bounding_boxes(bounding_boxes: torch.Tensor, format: Optional[tv_tensors.BoundingBoxFormat] = ..., canvas_size: Optional[Tuple[int, int]] = ..., min_size: float = ...) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove degenerate/invalid bounding boxes and return the corresponding indexing mask.

    This removes bounding boxes that:

    - are below a given ``min_size``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :func:`~torchvision.transforms.v2.functional.clamp_bounding_boxes` first to avoid undesired removals.

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        bounding_boxes (Tensor or :class:`~torchvision.tv_tensors.BoundingBoxes`): The bounding boxes to be sanitized.
        format (str or :class:`~torchvision.tv_tensors.BoundingBoxFormat`, optional): The format of the bounding boxes.
            Must be left to none if ``bounding_boxes`` is a :class:`~torchvision.tv_tensors.BoundingBoxes` object.
        canvas_size (tuple of int, optional): The canvas_size of the bounding boxes
            (size of the corresponding image/video).
            Must be left to none if ``bounding_boxes`` is a :class:`~torchvision.tv_tensors.BoundingBoxes` object.
        min_size (float, optional) The size below which bounding boxes are removed. Default is 1.

    Returns:
        out (tuple of Tensors): The subset of valid bounding boxes, and the corresponding indexing mask.
        The mask can then be used to subset other tensors (e.g. labels) that are associated with the bounding boxes.
    """
    ...
