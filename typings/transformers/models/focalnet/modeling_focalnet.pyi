"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...utils.backbone_utils import BackboneMixin
from .configuration_focalnet import FocalNetConfig

""" PyTorch FocalNet model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
@dataclass
class FocalNetEncoderOutput(ModelOutput):
    """
    FocalNet encoder's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class FocalNetModelOutput(ModelOutput):
    """
    FocalNet model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    last_hidden_state: torch.FloatTensor = ...
    pooler_output: Optional[torch.FloatTensor] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class FocalNetMaskedImageModelingOutput(ModelOutput):
    """
    FocalNet masked image model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    loss: Optional[torch.FloatTensor] = ...
    reconstruction: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class FocalNetImageClassifierOutput(ModelOutput):
    """
    FocalNet outputs for image classification.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...


class FocalNetEmbeddings(nn.Module):
    """
    Construct the patch embeddings and layernorm. Optionally, also the mask token.
    """
    def __init__(self, config, use_mask_token=...) -> None:
        ...

    def forward(self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = ...) -> Tuple[torch.Tensor]:
        ...



class FocalNetPatchEmbeddings(nn.Module):
    def __init__(self, config, image_size, patch_size, num_channels, embed_dim, add_norm=..., use_conv_embed=..., is_stem=...) -> None:
        ...

    def maybe_pad(self, pixel_values, height, width): # -> Tensor:
        ...

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        ...



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

class FocalNetDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    def extra_repr(self) -> str:
        ...



class FocalNetModulation(nn.Module):
    def __init__(self, config, index, dim, focal_factor=..., bias=..., projection_dropout=...) -> None:
        ...

    def forward(self, hidden_state): # -> Any:
        """
        Args:
            hidden_state:
                Input features with shape of (batch_size, height, width, num_channels)
        """
        ...



class FocalNetMlp(nn.Module):
    def __init__(self, config, in_features, hidden_features=..., out_features=..., drop=...) -> None:
        ...

    def forward(self, hidden_state): # -> Any:
        ...



class FocalNetLayer(nn.Module):
    r"""Focal Modulation Network layer (block).

    Args:
        config (`FocalNetConfig`):
            Model config.
        index (`int`):
            Layer index.
        dim (`int`):
            Number of input channels.
        input_resolution (`Tuple[int]`):
            Input resulotion.
        drop_path (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
    """
    def __init__(self, config, index, dim, input_resolution, drop_path=...) -> None:
        ...

    def forward(self, hidden_state, input_dimensions):
        ...



class FocalNetStage(nn.Module):
    def __init__(self, config, index, input_resolution) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int]) -> Tuple[torch.Tensor]:
        ...



class FocalNetEncoder(nn.Module):
    def __init__(self, config, grid_size) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], output_hidden_states: Optional[bool] = ..., output_hidden_states_before_downsampling: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, FocalNetEncoderOutput]:
        ...



class FocalNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FocalNetConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


FOCALNET_START_DOCSTRING = ...
FOCALNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare FocalNet Model outputting raw hidden-states without any specific head on top.", FOCALNET_START_DOCSTRING)
class FocalNetModel(FocalNetPreTrainedModel):
    def __init__(self, config, add_pooling_layer=..., use_mask_token=...) -> None:
        ...

    def get_input_embeddings(self): # -> FocalNetPatchEmbeddings:
        ...

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=FocalNetModelOutput, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., bool_masked_pos: Optional[torch.BoolTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, FocalNetModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        ...



@add_start_docstrings("""FocalNet Model with a decoder on top for masked image modeling.

    This follows the same implementation as in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """, FOCALNET_START_DOCSTRING)
class FocalNetForMaskedImageModeling(FocalNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FocalNetMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., bool_masked_pos: Optional[torch.BoolTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, FocalNetMaskedImageModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, FocalNetConfig, FocalNetForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-base-simmim-window6-192")
        >>> config = FocalNetConfig()
        >>> model = FocalNetForMaskedImageModeling(config)

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.logits
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 192, 192]
        ```"""
        ...



@add_start_docstrings("""
    FocalNet Model with an image classification head on top (a linear layer on top of the pooled output) e.g. for
    ImageNet.
    """, FOCALNET_START_DOCSTRING)
class FocalNetForImageClassification(FocalNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=FocalNetImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, FocalNetImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



@add_start_docstrings("""
    FocalNet backbone, to be used with frameworks like X-Decoder.
    """, FOCALNET_START_DOCSTRING)
class FocalNetBackbone(FocalNetPreTrainedModel, BackboneMixin):
    def __init__(self, config: FocalNetConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> BackboneOutput:
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

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-tiny-lrf")
        >>> model = AutoBackbone.from_pretrained("microsoft/focalnet-tiny-lrf")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        ...

