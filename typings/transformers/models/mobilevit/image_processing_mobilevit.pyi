"""
This type stub file was generated by pyright.
"""

import numpy as np
import PIL
from typing import Dict, List, Optional, Tuple, Union
from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling
from ...utils import TensorType, is_torch_available, is_vision_available

"""Image processor class for MobileViT."""
if is_vision_available():
    ...
if is_torch_available():
    ...
logger = ...
class MobileViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MobileViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Controls the size of the output image after resizing. Can be overridden by the `size` parameter in the
            `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter
            in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped. Can be overridden by the `do_center_crop` parameter in
            the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Desired output size `(size["height"], size["width"])` when applying center-cropping. Can be overridden by
            the `crop_size` parameter in the `preprocess` method.
        do_flip_channel_order (`bool`, *optional*, defaults to `True`):
            Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order`
            parameter in the `preprocess` method.
    """
    model_input_names = ...
    def __init__(self, do_resize: bool = ..., size: Dict[str, int] = ..., resample: PILImageResampling = ..., do_rescale: bool = ..., rescale_factor: Union[int, float] = ..., do_center_crop: bool = ..., crop_size: Dict[str, int] = ..., do_flip_channel_order: bool = ..., **kwargs) -> None:
        ...

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling = ..., data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        ...

    def flip_channel_order(self, image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]] = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ...) -> np.ndarray:
        """
        Flip the color channels from RGB to BGR or vice versa.

        Args:
            image (`np.ndarray`):
                The image, represented as a numpy array.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        ...

    def __call__(self, images, segmentation_maps=..., **kwargs): # -> BatchFeature:
        """
        Preprocesses a batch of images and optionally segmentation maps.

        Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
        passed in as positional arguments.
        """
        ...

    def preprocess(self, images: ImageInput, segmentation_maps: Optional[ImageInput] = ..., do_resize: bool = ..., size: Dict[str, int] = ..., resample: PILImageResampling = ..., do_rescale: bool = ..., rescale_factor: float = ..., do_center_crop: bool = ..., crop_size: Dict[str, int] = ..., do_flip_channel_order: bool = ..., return_tensors: Optional[Union[str, TensorType]] = ..., data_format: ChannelDimension = ..., input_data_format: Optional[Union[str, ChannelDimension]] = ..., **kwargs) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation map to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image by rescale factor.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop if `do_center_crop` is set to `True`.
            do_flip_channel_order (`bool`, *optional*, defaults to `self.do_flip_channel_order`):
                Whether to flip the channel order of the image.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        ...

    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = ...): # -> list[Any]:
        """
        Converts the output of [`MobileViTForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileViTForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        ...

