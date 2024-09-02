"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_hubert import HubertConfig

""" PyTorch Hubert model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_CTC_EXPECTED_OUTPUT = ...
_CTC_EXPECTED_LOSS = ...
_SEQ_CLASS_CHECKPOINT = ...
_SEQ_CLASS_EXPECTED_OUTPUT = ...
_SEQ_CLASS_EXPECTED_LOSS = ...
class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class HubertLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states):
        ...



class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings) -> None:
        ...

    def forward(self, hidden_states):
        ...



class HubertFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_values): # -> Any:
        ...



class HubertFeatureExtractor(HubertFeatureEncoder):
    def __init__(self, config) -> None:
        ...



class HubertFeatureProjection(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class HubertAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., is_causal: bool = ..., config: Optional[HubertConfig] = ...) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...



class HubertFeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class HubertEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., output_attentions=...): # -> tuple[Any, Any] | tuple[Any]:
        ...



class HubertAttnAdapterLayer(nn.Module):
    def __init__(self, config) -> None:
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        ...

    def forward(self, hidden_states: torch.FloatTensor): # -> FloatTensor:
        ...



class HubertEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...



class HubertEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.tensor, attention_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | BaseModelOutput:
        ...



class HubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        ...



class HubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = HubertConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


HUBERT_START_DOCSTRING = ...
HUBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Hubert Model transformer outputting raw hidden-states without any specific head on top.", HUBERT_START_DOCSTRING)
class HubertModel(HubertPreTrainedModel):
    def __init__(self, config: HubertConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., mask_time_indices: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, HubertModel
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""
        ...



@add_start_docstrings("""Hubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""", HUBERT_START_DOCSTRING)
class HubertForCTC(HubertPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = ...) -> None:
        ...

    def tie_weights(self): # -> None:
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """
        ...

    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC, expected_output=_CTC_EXPECTED_OUTPUT, expected_loss=_CTC_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...



@add_start_docstrings("""
    Hubert Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """, HUBERT_START_DOCSTRING)
class HubertForSequenceClassification(HubertPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        ...

    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...

    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_SEQ_CLASS_CHECKPOINT, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_SEQ_CLASS_EXPECTED_OUTPUT, expected_loss=_SEQ_CLASS_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

