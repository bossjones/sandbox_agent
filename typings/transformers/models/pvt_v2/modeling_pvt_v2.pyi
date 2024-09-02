"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...utils.backbone_utils import BackboneMixin
from .configuration_pvt_v2 import PvtV2Config

"""PyTorch PVTv2 model."""
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

class PvtV2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    def extra_repr(self) -> str:
        ...



class PvtV2OverlapPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, config: PvtV2Config, layer_idx: int) -> None:
        ...

    def forward(self, pixel_values): # -> tuple[Any, Any, Any]:
        ...



class PvtV2DepthWiseConv(nn.Module):
    """
    Depth-wise (DW) convolution to infuse positional information using zero-padding. Depth-wise convolutions
    have an equal number of groups to the number of input channels, meaning one filter per input channel. This
    reduces the overall parameters and compute costs since the key purpose of this layer is position encoding.
    """
    def __init__(self, config: PvtV2Config, dim: int = ...) -> None:
        ...

    def forward(self, hidden_states, height, width): # -> Any:
        ...



class PvtV2SelfAttention(nn.Module):
    """Efficient self-attention mechanism."""
    def __init__(self, config: PvtV2Config, hidden_size: int, num_attention_heads: int, spatial_reduction_ratio: int) -> None:
        ...

    def transpose_for_scores(self, hidden_states) -> torch.Tensor:
        ...

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = ...) -> Tuple[torch.Tensor]:
        ...

    def prune_heads(self, heads): # -> None:
        ...



class PvtV2ConvFeedForwardNetwork(nn.Module):
    def __init__(self, config: PvtV2Config, in_features: int, hidden_features: Optional[int] = ..., out_features: Optional[int] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, height, width) -> torch.Tensor:
        ...



class PvtV2BlockLayer(nn.Module):
    def __init__(self, config: PvtV2Config, layer_idx: int, drop_path: float = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = ...): # -> Any:
        ...



class PvtV2EncoderLayer(nn.Module):
    def __init__(self, config: PvtV2Config, layer_idx: int) -> None:
        ...

    def forward(self, hidden_states, output_attentions): # -> tuple[tuple[Any, tuple[()] | Any | None] | tuple[Any], Any, Any]:
        ...



class PvtV2Encoder(nn.Module):
    def __init__(self, config: PvtV2Config) -> None:
        ...

    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...



class PvtV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = PvtV2Config
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


PVT_V2_START_DOCSTRING = ...
PVT_V2_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Pvt-v2 encoder outputting raw hidden-states without any specific head on top.", PVT_V2_START_DOCSTRING)
class PvtV2Model(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config) -> None:
        ...

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...



@add_start_docstrings("""
    Pvt-v2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """, PVT_V2_START_DOCSTRING)
class PvtV2ForImageClassification(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config) -> None:
        ...

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.Tensor], labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



@add_start_docstrings("""
    PVTv2 backbone, to be used with frameworks like DETR and MaskFormer.
    """, PVT_V2_START_DOCSTRING)
class PvtV2Backbone(PvtV2Model, BackboneMixin):
    def __init__(self, config: PvtV2Config) -> None:
        ...

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
        >>> model = AutoBackbone.from_pretrained(
        ...     "OpenGVLab/pvt_v2_b0", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 256, 7, 7]
        ```"""
        ...

