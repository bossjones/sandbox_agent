"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Any, Optional, Set, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_flava import FlavaConfig, FlavaImageCodebookConfig, FlavaImageConfig, FlavaMultimodalConfig, FlavaTextConfig

""" PyTorch FLAVA model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CHECKPOINT_FOR_CODEBOOK_DOC = ...
_CONFIG_CLASS_FOR_IMAGE_MODEL_DOC = ...
_CONFIG_CLASS_FOR_TEXT_MODEL_DOC = ...
_CONFIG_CLASS_FOR_MULTIMODAL_MODEL_DOC = ...
_EXPECTED_IMAGE_OUTPUT_SHAPE = ...
FLAVA_CODEBOOK_PRETRAINED_MODEL_ARCHIVE_LIST = ...
LOGIT_SCALE_CLAMP_MIN = ...
LOGIT_SCALE_CLAMP_MAX = ...
FlavaPossibleConfigs = Union[FlavaTextConfig, FlavaImageConfig, FlavaMultimodalConfig]
@dataclass
class FlavaModelOutput(ModelOutput):
    """
    Output from FlavaModel containing embeddings and outputs from individual encoders.

    Note that `image_embeddings` and `text_embeddigns` returned are similar to pooled output returned from a
    transformer. If you want embeddings for contrastive loss or retrieval use a FLAVA model's `image_projection` and
    `text_projection` layers on `image_embeddings` and `text_embeddings` respectively.

    Args:
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present):
            The image embeddings which are basically the pooled output of [`FlavaImageModel`].
        image_output (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present):
            The output of the [`FlavaImageModel`].
        text_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` are present):
            The text embeddings which are basically the pooled output of [`FlavaTextModel`].
        text_output (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids` are present):
            The output of the [`FlavaTextModel`].
        multimodal_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present and `skip_multimodal_encoder` is `None` or `False`):
            The multimodal embeddings which are basically the pooled output of [`FlavaTextModel`].
        multimodal_output (`BaseModelOutputWithPooling`, returned when `input_ids` and `pixel_values` are present and `skip_multimodal_encoder` is `None` or `False`):
            The output of the [`FlavaMultimodalModel`].
    """
    image_embeddings: Optional[torch.FloatTensor] = ...
    image_output: Optional[BaseModelOutputWithPooling] = ...
    text_embeddings: Optional[torch.FloatTensor] = ...
    text_output: Optional[BaseModelOutputWithPooling] = ...
    multimodal_embeddings: Optional[torch.FloatTensor] = ...
    multimodal_output: Optional[BaseModelOutputWithPooling] = ...
    def to_tuple(self) -> Tuple[Any]:
        ...



@dataclass
class FlavaLosses(ModelOutput):
    """Class representing pretraining losses from FLAVA model

    Args:
        mim (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mim_labels` and `pixel_values` are present, `input_ids_masked` is absent and `mim_weight` > 0.:
            Masked Image Modeling loss as used in BeIT calculated only for unimodal image data.
        mlm (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mlm_labels` and `input_ids_masked` are present, `pixel_values` is absent and `mlm_weight` > 0.:
            Masked Language Modeling loss as used in BERT calculated only for unimodal text data.
        itm (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `itm_labels`, `input_ids_masked`, `pixel_values` are present and `itm_weight` > 0.:
            Image Text Matching (ITM) loss calculated for paired image-text data. Note that ITM loss is calculated on
            masked pairs in FLAVA.
        global_contrastive (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `input_ids` and `pixel_values` are present and `global_contrastive_weight` > 0.:
            Contrastive loss for image-text similarity similar to CLIP but calculated globally for paired image-text
            data. This is calculated on unmasked images and texts.
        mmm_image (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mim_labels`, `pixel_values` and `input_ids_masked` are present and `mmm_image_weight` > 0.:
            Masked Multimodal Modeling loss's image component calculated on paired image-text data.
        mmm_text (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mlm_labels`, `pixel_values` and `input_ids_masked` are present and `mmm_text_weight` > 0.:
            Masked Multimodal Modeling loss's text component calculated on paired image-text data.
    """
    mim: Optional[torch.FloatTensor] = ...
    mlm: Optional[torch.FloatTensor] = ...
    itm: Optional[torch.FloatTensor] = ...
    global_contrastive: Optional[torch.FloatTensor] = ...
    mmm_image: Optional[torch.FloatTensor] = ...
    mmm_text: Optional[torch.FloatTensor] = ...
    def all_none(self) -> bool:
        ...



@dataclass
class FlavaForPreTrainingOutput(ModelOutput):
    """
    Output from FlavaForPreTraining containing embeddings, and outputs from individual encoders.

    Note that `image_embeddings` and `text_embeddings` returned are similar to pooled output returned from a
    transformer. If you want embeddings for contrastive loss or retrieval use a FLAVA model's `image_projection` and
    `text_projection` layers on `image_embeddings` and `text_embeddings` respectively.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `return_loss` is True):
            Total loss calculated for this model.
        loss_info (`FlavaLosses`):
            Detailed info for FLAVA Pretraining losses. Check `FlavaLosses` class description for the information on
            the keys.
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present):
            The image embeddings which are basically the pooled output of [`FlavaImageModel`].
        image_output (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present):
            The output of the [`FlavaImageModel`].
        text_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` are present):
            The text embeddings which are basically the pooled output of [`FlavaTextModel`].
        text_output (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids` are present):
            The output of the [`FlavaTextModel`].
        multimodal_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`):
            The multimodal embeddings which are basically the pooled output of [`FlavaTextModel`].
        multimodal_output (`BaseModelOutputWithPooling`, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`):
            The output of the [`FlavaMultimodalModel`].

        image_masked_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present):
            The image embeddings which are basically the pooled output of [`FlavaImageModel`]. Uses `bool_masked_pos`
            to create masked images.
        image_masked_output (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present):
            The output of the [`FlavaImageModel`]. Uses `bool_masked_pos` to create masked images.
        text_masked_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids_masked` are present):
            The text embeddings which are basically the pooled output of [`FlavaTextModel`].
        text_masked_output (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` are present):
            The output of the [`FlavaTextModel`].
        multimodal_masked_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present):
            The multimodal embeddings which are basically the pooled output of [`FlavaTextModel`].
        multimodal_masked_output (`BaseModelOutputWithPooling`, returned when `input_ids_masked` and `pixel_values` are present):
            The output of the [`FlavaMultimodalModel`].

        mim_logits (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape `(total_masked_patches, image_vocab_size)` , *optional*, returned when `pixel_values` are present and `input_ids_masked` are not):
                The logits for MIM unimodal loss. Uses `book_masked_pos` to get masked patches. The flattened output is
                returned when `bool_masked_pos` has some of the patches masked.
        mlm_logits (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(total_masked_seq_length, text_vocab_size)`, *optional*, returned when `input_ids_masked` are present and `pixel_values` are not):
                The logits for MLM unimodal loss. The flattened output is returned when `input_ids_masked` has some of
                the tokens masked.
        itm_logits (`torch.FloatTensor` of shape `(batch_size, 2)`, *optional*, returned when `input_ids_masked` and `pixel_values` are present):
                The logits for ITM loss. Note that ITM loss is calculated on masked pairs in FLAVA.
        mmm_image_logits (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape`(total_masked_patches, image_vocab_size)`, *optional*, returned when `pixel_values` and `input_ids_masked` are present):
                The logits for MMM image multimodal loss. Uses `book_masked_pos` to get masked patches. The flattened
                output is returned when `bool_masked_pos` has some of the patches masked.
        mmm_text_logits (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(`(total_masked_seq_length, text_vocab_size)`), *optional*, returned when `pixel_values` and `input_ids_masked` are present):
                The logits for MMM text multimodal loss. The flattened output is returned when `input_ids_masked` has
                some of the tokens masked.
        contrastive_logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeddings` and `text_embeddings` but passed through FLAVA's
            `image_projection` and `text_projection` layers respectively. This represents the image-text similarity
            scores. This is calculated on unmasked images and texts.
        contrastive_logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeddings` and `image_embeddings` but passed through FLAVA's
            `text_projection` and `image_projection` layers respectively. This is calculated on unmasked images and
            texts.
    """
    loss: Optional[torch.FloatTensor] = ...
    loss_info: FlavaLosses = ...
    image_embeddings: Optional[torch.FloatTensor] = ...
    image_output: Optional[BaseModelOutputWithPooling] = ...
    text_embeddings: Optional[torch.FloatTensor] = ...
    text_output: Optional[BaseModelOutputWithPooling] = ...
    multimodal_embeddings: Optional[torch.FloatTensor] = ...
    multimodal_output: Optional[BaseModelOutputWithPooling] = ...
    image_masked_embeddings: Optional[torch.FloatTensor] = ...
    image_masked_output: Optional[BaseModelOutputWithPooling] = ...
    text_masked_embeddings: Optional[torch.FloatTensor] = ...
    text_masked_output: Optional[BaseModelOutputWithPooling] = ...
    multimodal_masked_embeddings: Optional[torch.FloatTensor] = ...
    multimodal_masked_output: Optional[BaseModelOutputWithPooling] = ...
    mim_logits: Optional[torch.FloatTensor] = ...
    mlm_logits: Optional[torch.FloatTensor] = ...
    itm_logits: Optional[torch.FloatTensor] = ...
    contrastive_logits_per_image: Optional[torch.FloatTensor] = ...
    contrastive_logits_per_text: Optional[torch.FloatTensor] = ...
    mmm_image_logits: Optional[torch.FloatTensor] = ...
    mmm_text_logits: Optional[torch.FloatTensor] = ...
    def to_tuple(self) -> Tuple[Any]:
        ...



class FlavaImageEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """
    def __init__(self, config: FlavaImageConfig, use_mask_token: bool = ...) -> None:
        ...

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/image_transformer.py#L174
        """
        ...

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = ..., interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...



class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, image_size: int = ..., patch_size: Union[int, Tuple[int, int]] = ..., num_channels: int = ..., embed_dim: int = ...) -> None:
        ...

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = ...) -> torch.Tensor:
        ...



class FlavaTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_ids: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ...): # -> Any:
        ...



class FlavaSelfAttention(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        ...

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class FlavaSelfOutput(nn.Module):
    """
    The residual connection is defined in FlavaLayer (same as ViTLayer) instead of here (as is the case with other
    models), due to the layernorm applied before each block.
    """
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class FlavaAttention(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        ...

    def prune_heads(self, heads: Set[int]) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class FlavaIntermediate(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class FlavaOutput(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class FlavaLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        ...



class FlavaEncoder(nn.Module):
    def __init__(self, config: FlavaConfig) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[tuple, BaseModelOutput]:
        ...



class FlavaPooler(nn.Module):
    def __init__(self, config: FlavaPossibleConfigs) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor): # -> Any:
        ...



FLAVA_START_DOCSTRING = ...
FLAVA_INPUTS_DOCSTRING_COMMON = ...
FLAVA_IMAGE_INPUTS_DOCSTRING_BASE = ...
FLAVA_IMAGE_INPUTS_DOCSTRING = ...
FLAVA_TEXT_INPUTS_DOCSTRING_BASE = ...
FLAVA_TEXT_INPUTS_DOCSTRING = ...
FLAVA_MULTIMODAL_INPUTS_DOCSTRING = ...
FLAVA_MODEL_INPUTS_DOCSTRING_BASE = ...
FLAVA_MODEL_INPUTS_DOCSTRING = ...
FLAVA_PRETRAINING_INPUTS_DOCSTRING = ...
FLAVA_PRETRAINING_START_DOCSTRING_EXTRA = ...
class FlavaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FlavaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...


@add_start_docstrings("The bare FLAVA Image Model transformer outputting raw hidden-states without any specific head on top.", FLAVA_START_DOCSTRING.format(config="FlavaImageConfig"))
class FlavaImageModel(FlavaPreTrainedModel):
    config_class = FlavaImageConfig
    base_model_prefix = ...
    main_input_name = ...
    def __init__(self, config: FlavaImageConfig, add_pooling_layer: bool = ...) -> None:
        ...

    def get_input_embeddings(self) -> nn.Module:
        ...

    def set_input_embeddings(self, value: nn.Module): # -> None:
        ...

    @add_start_docstrings_to_model_forward(FLAVA_IMAGE_INPUTS_DOCSTRING.format("batch_size, image_num_patches"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_CLASS_FOR_IMAGE_MODEL_DOC, modality="vision", expected_output=_EXPECTED_IMAGE_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor] = ..., bool_masked_pos: Optional[torch.BoolTensor] = ..., interpolate_pos_encoding: Optional[bool] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, BaseModelOutputWithPooling]:
        ...



@add_start_docstrings("The bare FLAVA Text Model transformer outputting raw hidden-states without any specific head on top.", FLAVA_START_DOCSTRING.format(config="FlavaTextConfig"))
class FlavaTextModel(FlavaPreTrainedModel):
    config_class = FlavaTextConfig
    base_model_prefix = ...
    def __init__(self, config: FlavaTextConfig, add_pooling_layer: bool = ...) -> None:
        ...

    def get_input_embeddings(self) -> PatchEmbeddings:
        ...

    def set_input_embeddings(self, value: nn.Module): # -> None:
        ...

    @add_start_docstrings_to_model_forward(FLAVA_TEXT_INPUTS_DOCSTRING.format("batch_size, text_seq_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_CLASS_FOR_TEXT_MODEL_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, BaseModelOutputWithPooling]:
        ...



@add_start_docstrings("The bare FLAVA Multimodal Model transformer outputting raw hidden-states without any specific head on top.", FLAVA_START_DOCSTRING.format(config="FlavaMultimodalConfig"))
class FlavaMultimodalModel(FlavaPreTrainedModel):
    config_class = FlavaMultimodalConfig
    base_model_prefix = ...
    main_input_name = ...
    def __init__(self, config: FlavaMultimodalConfig, add_pooling_layer=...) -> None:
        ...

    @add_start_docstrings_to_model_forward(FLAVA_MULTIMODAL_INPUTS_DOCSTRING.format("batch_size, image_num_patches + text_seq_len"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_CLASS_FOR_MULTIMODAL_MODEL_DOC)
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[tuple, BaseModelOutputWithPooling]:
        ...



@add_start_docstrings("The bare FLAVA Model transformer outputting raw hidden-states without any specific head on top.", FLAVA_START_DOCSTRING.format(config="FlavaConfig"))
class FlavaModel(FlavaPreTrainedModel):
    config_class = FlavaConfig
    def __init__(self, config: FlavaConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(FLAVA_TEXT_INPUTS_DOCSTRING.format("batch_size, text_seq_length"))
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        ...

    @add_start_docstrings_to_model_forward(FLAVA_IMAGE_INPUTS_DOCSTRING.format("batch_size, image_num_patches"))
    def get_image_features(self, pixel_values: Optional[torch.Tensor] = ..., bool_masked_pos: Optional[torch.BoolTensor] = ..., interpolate_pos_encoding: Optional[bool] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        ...

    @add_start_docstrings_to_model_forward(FLAVA_MODEL_INPUTS_DOCSTRING.format("batch_size, image_num_patches + text_seq_len"))
    @replace_return_docstrings(output_type=FlavaModelOutput, config_class=FlavaConfig)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., bool_masked_pos: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., image_attention_mask: Optional[torch.Tensor] = ..., skip_multimodal_encoder: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: bool = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, FlavaOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FlavaModel

        >>> model = FlavaModel.from_pretrained("facebook/flava-full")
        >>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)

        >>> image_embeddings = outputs.image_embeddings
        >>> text_embeddings = outputs.text_embeddings
        >>> multimodal_embeddings = outputs.multimodal_embeddings

        >>> outputs.image_embeddings.shape
        torch.Size([1, 197, 768])

        >>> text_embeddings.shape
        torch.Size([1, 7, 768])

        >>> multimodal_embeddings.shape
        torch.Size([1, 205, 768])
        ```
        """
        ...



class FlavaImageCodebookResPath(nn.Module):
    def __init__(self, in_size: int, out_size: int, **kwargs) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...



class FlavaImageCodebookBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_layers: int, **kwargs) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...



class FlavaImageCodebookLayerGroup(nn.Module):
    def __init__(self, num_blocks: int, num_layers: int, in_size: int, out_size: int, use_pool: bool = ...) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...



@add_start_docstrings("""
    The FLAVA's image codebook model inspired from DALL-E's original encoder. Outputs raw hidden states and can be used
    to generate image tokens for an image based on DALL-E's vocab. Used to generate labels for MIM. Use
    `get_codebook_indices` to get image tokens for an image.
    """, FLAVA_START_DOCSTRING.format(config="FlavaImageCodebookConfig"))
class FlavaImageCodebook(FlavaPreTrainedModel):
    base_model_prefix = ...
    config_class = FlavaImageCodebookConfig
    main_input_name = ...
    supports_gradient_checkpointing = ...
    def __init__(self, config: FlavaImageCodebookConfig, **kwargs: Any) -> None:
        ...

    def get_codebook_indices(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ...

    def get_codebook_probs(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        ...



class FlavaPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class FlavaMaskedPredictionHead(nn.Module):
    def __init__(self, config, weight=...) -> None:
        ...

    def forward(self, x): # -> Any:
        ...



class FlavaITMHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, x): # -> Any:
        ...



class FlavaGlobalContrastiveHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, image_embeddings, text_embeddings, logit_scale): # -> tuple[Tensor, Tensor, Tensor | Any]:
        ...



@add_start_docstrings("""
    The FLAVA model for pretraining which outputs losses, embeddings, logits and transformer outputs.
    """, FLAVA_START_DOCSTRING.format(config="FlavaConfig") + FLAVA_PRETRAINING_START_DOCSTRING_EXTRA)
class FlavaForPreTraining(FlavaPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: FlavaConfig, image_codebook: Optional[nn.Module] = ...) -> None:
        ...

    @add_start_docstrings_to_model_forward(FLAVA_PRETRAINING_INPUTS_DOCSTRING.format("batch_size, text_seq_len", "batch_size, image_num_patches"))
    @replace_return_docstrings(output_type=FlavaForPreTrainingOutput, config_class=FlavaConfig)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., input_ids_masked: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., codebook_pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., bool_masked_pos: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., image_attention_mask: Optional[torch.Tensor] = ..., skip_unmasked_multimodal_encoder: bool = ..., mlm_labels: Optional[torch.Tensor] = ..., mim_labels: Optional[torch.Tensor] = ..., itm_labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: bool = ..., return_dict: Optional[bool] = ..., return_loss: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], FlavaForPreTrainingOutput]:
        """
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import FlavaForPreTraining, AutoProcessor

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
        >>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

        >>> text = ["a photo of a cat"]

        >>> inputs = processor(
        ...     images=[image],
        ...     text=text,
        ...     return_masks=True,
        ...     return_codebook_pixels=True,
        ...     padding=True,
        ...     max_length=77,
        ...     return_tensors="pt",
        ... )


        >>> output = model(**inputs)
        ```

        Return:

        """
        ...

