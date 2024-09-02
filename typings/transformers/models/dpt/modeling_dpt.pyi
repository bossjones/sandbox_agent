"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
from torch import nn
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from .configuration_dpt import DPTConfig

""" PyTorch DPT (Dense Prediction Transformers) model.

This implementation is heavily inspired by OpenMMLab's implementation, found here:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/dpt_head.py.

"""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
@dataclass
class BaseModelOutputWithIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains intermediate activations that can be used at later stages. Useful
    in the context of Vision models.:

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """
    last_hidden_states: torch.FloatTensor = ...
    intermediate_activations: Optional[Tuple[torch.FloatTensor, ...]] = ...


@dataclass
class BaseModelOutputWithPoolingAndIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states as well as intermediate
    activations that can be used by the model at later stages.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """
    last_hidden_state: torch.FloatTensor = ...
    pooler_output: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = ...
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = ...
    intermediate_activations: Optional[Tuple[torch.FloatTensor, ...]] = ...


class DPTViTHybridEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config, feature_size=...) -> None:
        ...

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ..., return_dict: bool = ...) -> torch.Tensor:
        ...



class DPTViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """
    def __init__(self, config) -> None:
        ...

    def forward(self, pixel_values, return_dict=...): # -> tuple[Any] | BaseModelOutputWithIntermediateActivations:
        ...



class DPTViTPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """
    def __init__(self, config) -> None:
        ...

    def forward(self, pixel_values): # -> Any:
        ...



class DPTViTSelfAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        ...

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class DPTViTSelfOutput(nn.Module):
    """
    The residual connection is defined in DPTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config: DPTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class DPTViTAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        ...

    def prune_heads(self, heads: Set[int]) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class DPTViTIntermediate(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class DPTViTOutput(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class DPTViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: DPTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class DPTViTEncoder(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[tuple, BaseModelOutput]:
        ...



class DPTReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
       `config.readout_type`.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: List[torch.Tensor], patch_height=..., patch_width=...) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        ...



class DPTReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor) -> None:
        ...

    def forward(self, hidden_state): # -> Any:
        ...



class DPTFeatureFusionStage(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> list[Any]:
        ...



class DPTPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        ...



class DPTFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
        align_corners (`bool`, *optional*, defaults to `True`):
            The align_corner setting for bilinear upsample.
    """
    def __init__(self, config, align_corners=...) -> None:
        ...

    def forward(self, hidden_state, residual=...): # -> Any:
        ...



class DPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DPTConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


DPT_START_DOCSTRING = ...
DPT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DPT Model transformer outputting raw hidden-states without any specific head on top.", DPT_START_DOCSTRING)
class DPTModel(DPTPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...

    def get_input_embeddings(self): # -> DPTViTHybridEmbeddings | DPTViTEmbeddings | Any | DPTViTPatchEmbeddings:
        ...

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndIntermediateActivations, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor, head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPoolingAndIntermediateActivations]:
        ...



class DPTViTPooler(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class DPTNeck(nn.Module):
    """
    DPTNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DPT, it includes 2 stages:

    * DPTReassembleStage
    * DPTFeatureFusionStage.

    Args:
        config (dict): config dict.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: List[torch.Tensor], patch_height=..., patch_width=...) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        ...



class DPTDepthEstimationHead(nn.Module):
    """
    Output head head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    supplementary material).
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        ...



@add_start_docstrings("""
    DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """, DPT_START_DOCSTRING)
class DPTForDepthEstimation(DPTPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, head_mask: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        ...



class DPTSemanticSegmentationHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        ...



class DPTAuxiliaryHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



@add_start_docstrings("""
    DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """, DPT_START_DOCSTRING)
class DPTForSemanticSegmentation(DPTPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, DPTForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large-ade")
        >>> model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""
        ...

