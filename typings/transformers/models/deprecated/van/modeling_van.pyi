"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ....modeling_outputs import BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_van import VanConfig

""" PyTorch Visual Attention Network (VAN) model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
def drop_path(input: torch.Tensor, drop_prob: float = ..., training: bool = ...) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    ...

class VanDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    def extra_repr(self) -> str:
        ...



class VanOverlappingPatchEmbedder(nn.Module):
    """
    Downsamples the input using a patchify operation with a `stride` of 4 by default making adjacent windows overlap by
    half of the area. From [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = ..., stride: int = ...) -> None:
        ...

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...



class VanMlpLayer(nn.Module):
    """
    MLP with depth-wise convolution, from [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int, hidden_act: str = ..., dropout_rate: float = ...) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class VanLargeKernelAttention(nn.Module):
    """
    Basic Large Kernel Attention (LKA).
    """
    def __init__(self, hidden_size: int) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class VanLargeKernelAttentionLayer(nn.Module):
    """
    Computes attention using Large Kernel Attention (LKA) and attends the input.
    """
    def __init__(self, hidden_size: int) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class VanSpatialAttentionLayer(nn.Module):
    """
    Van spatial attention layer composed by projection (via conv) -> act -> Large Kernel Attention (LKA) attention ->
    projection (via conv) + residual connection.
    """
    def __init__(self, hidden_size: int, hidden_act: str = ...) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class VanLayerScaling(nn.Module):
    """
    Scales the inputs by a learnable parameter initialized by `initial_value`.
    """
    def __init__(self, hidden_size: int, initial_value: float = ...) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class VanLayer(nn.Module):
    """
    Van layer composed by normalization layers, large kernel attention (LKA) and a multi layer perceptron (MLP).
    """
    def __init__(self, config: VanConfig, hidden_size: int, mlp_ratio: int = ..., drop_path_rate: float = ...) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class VanStage(nn.Module):
    """
    VanStage, consisting of multiple layers.
    """
    def __init__(self, config: VanConfig, in_channels: int, hidden_size: int, patch_size: int, stride: int, depth: int, mlp_ratio: int = ..., drop_path_rate: float = ...) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class VanEncoder(nn.Module):
    """
    VanEncoder, consisting of multiple stages.
    """
    def __init__(self, config: VanConfig) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        ...



class VanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VanConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


VAN_START_DOCSTRING = ...
VAN_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare VAN model outputting raw features without any specific head on top. Note, VAN does not have an embedding" " layer.", VAN_START_DOCSTRING)
class VanModel(VanPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.FloatTensor], output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        ...



@add_start_docstrings("""
    VAN Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """, VAN_START_DOCSTRING)
class VanForImageClassification(VanPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

