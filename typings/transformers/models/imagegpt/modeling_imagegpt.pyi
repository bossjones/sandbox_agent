"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_imagegpt import ImageGPTConfig

"""PyTorch OpenAI ImageGPT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
def load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path):
    """
    Load tf checkpoints in a pytorch model
    """
    ...

class ImageGPTLayerNorm(nn.Module):
    def __init__(self, hidden_size: Tuple[int], eps: float = ...) -> None:
        ...

    def forward(self, tensor: torch.Tensor) -> tuple:
        ...



class ImageGPTAttention(nn.Module):
    def __init__(self, config, is_cross_attention: Optional[bool] = ..., layer_idx: Optional[int] = ...) -> None:
        ...

    def prune_heads(self, heads): # -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, layer_past: Optional[bool] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ...) -> tuple:
        ...



class ImageGPTMLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class ImageGPTBlock(nn.Module):
    def __init__(self, config, layer_idx=...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, layer_past: Optional[bool] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ...) -> tuple:
        ...



class ImageGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ImageGPTConfig
    load_tf_weights = ...
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    def __init__(self, *inputs, **kwargs) -> None:
        ...



IMAGEGPT_START_DOCSTRING = ...
IMAGEGPT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare ImageGPT Model transformer outputting raw hidden-states without any specific head on top.", IMAGEGPT_START_DOCSTRING)
class ImageGPTModel(ImageGPTPreTrainedModel):
    def __init__(self, config: ImageGPTConfig) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def set_input_embeddings(self, new_embeddings): # -> None:
        ...

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., **kwargs: Any) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ImageGPTModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
        >>> model = ImageGPTModel.from_pretrained("openai/imagegpt-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...



@add_start_docstrings("""
    The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, IMAGEGPT_START_DOCSTRING)
class ImageGPTForCausalImageModeling(ImageGPTPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: ImageGPTConfig) -> None:
        ...

    def get_output_embeddings(self): # -> Linear:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values: Optional[bool] = ..., **kwargs): # -> dict[str, Any]:
        ...

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., **kwargs: Any) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
        >>> import torch
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
        >>> model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> # unconditional generation of 8 images
        >>> batch_size = 4
        >>> context = torch.full((batch_size, 1), model.config.vocab_size - 1)  # initialize with SOS token
        >>> context = context.to(device)
        >>> output = model.generate(
        ...     input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40
        ... )

        >>> clusters = image_processor.clusters
        >>> height = image_processor.size["height"]
        >>> width = image_processor.size["width"]

        >>> samples = output[:, 1:].cpu().detach().numpy()
        >>> samples_img = [
        ...     np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [height, width, 3]).astype(np.uint8) for s in samples
        ... ]  # convert color cluster tokens back to pixels
        >>> f, axes = plt.subplots(1, batch_size, dpi=300)

        >>> for img, ax in zip(samples_img, axes):  # doctest: +IGNORE_RESULT
        ...     ax.axis("off")
        ...     ax.imshow(img)
        ```"""
        ...



@add_start_docstrings("""
    The ImageGPT Model transformer with an image classification head on top (linear layer).
    [`ImageGPTForImageClassification`] average-pools the hidden states in order to do the classification.
    """, IMAGEGPT_START_DOCSTRING)
class ImageGPTForImageClassification(ImageGPTPreTrainedModel):
    def __init__(self, config: ImageGPTConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., **kwargs: Any) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ImageGPTForImageClassification
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
        >>> model = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        ```"""
        ...

