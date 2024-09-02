"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from ..ops.misc import MLP
from ._api import WeightsEnum, register_model
from ._utils import handle_legacy_interface

__all__ = ["VisionTransformer", "ViT_B_16_Weights", "ViT_B_32_Weights", "ViT_L_16_Weights", "ViT_L_32_Weights", "ViT_H_14_Weights", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]
class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = ...
    activation_layer: Callable[..., nn.Module] = ...


class MLPBlock(MLP):
    """Transformer MLP block."""
    _version = ...
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float) -> None:
        ...



class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float, norm_layer: Callable[..., torch.nn.Module] = ...) -> None:
        ...

    def forward(self, input: torch.Tensor): # -> Any:
        ...



class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    def __init__(self, seq_length: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float, norm_layer: Callable[..., torch.nn.Module] = ...) -> None:
        ...

    def forward(self, input: torch.Tensor): # -> Any:
        ...



class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""
    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float = ..., attention_dropout: float = ..., num_classes: int = ..., representation_size: Optional[int] = ..., norm_layer: Callable[..., torch.nn.Module] = ..., conv_stem_configs: Optional[List[ConvStemConfig]] = ...) -> None:
        ...

    def forward(self, x: torch.Tensor): # -> Tensor:
        ...



_COMMON_META: Dict[str, Any] = ...
_COMMON_SWAG_META = ...
class ViT_B_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_SWAG_E2E_V1 = ...
    IMAGENET1K_SWAG_LINEAR_V1 = ...
    DEFAULT = ...


class ViT_B_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    IMAGENET1K_SWAG_E2E_V1 = ...
    IMAGENET1K_SWAG_LINEAR_V1 = ...
    DEFAULT = ...


class ViT_L_32_Weights(WeightsEnum):
    IMAGENET1K_V1 = ...
    DEFAULT = ...


class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = ...
    IMAGENET1K_SWAG_LINEAR_V1 = ...
    DEFAULT = ...


@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16(*, weights: Optional[ViT_B_16_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_32_Weights.IMAGENET1K_V1))
def vit_b_32(*, weights: Optional[ViT_B_32_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_b_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_32_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_16_Weights.IMAGENET1K_V1))
def vit_l_16(*, weights: Optional[ViT_L_16_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_16_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_L_32_Weights.IMAGENET1K_V1))
def vit_l_32(*, weights: Optional[ViT_L_32_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_l_32 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_L_32_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_L_32_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_L_32_Weights
        :members:
    """
    ...

@register_model()
@handle_legacy_interface(weights=("pretrained", None))
def vit_h_14(*, weights: Optional[ViT_H_14_Weights] = ..., progress: bool = ..., **kwargs: Any) -> VisionTransformer:
    """
    Constructs a vit_h_14 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_H_14_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_H_14_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_H_14_Weights
        :members:
    """
    ...

def interpolate_embeddings(image_size: int, patch_size: int, model_state: OrderedDict[str, torch.Tensor], interpolation_mode: str = ..., reset_heads: bool = ...) -> OrderedDict[str, torch.Tensor]:
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    ...
