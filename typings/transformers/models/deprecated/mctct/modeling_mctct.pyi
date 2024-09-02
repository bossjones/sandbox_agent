"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import PreTrainedModel
from .configuration_mctct import MCTCTConfig

""" PyTorch M-CTC-T model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_CTC_EXPECTED_OUTPUT = ...
_CTC_EXPECTED_LOSS = ...
class MCTCTConv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, input_features): # -> Tensor | Any:
        ...



class MCTCTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_features=..., token_type_ids=..., position_ids=..., inputs_embeds=..., past_key_values_length=...): # -> Any:
        ...



class MCTCTSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...

    def transpose_for_scores(self, x):
        ...

    def reshape_fortran(self, x, shape):
        ...

    def relative_position_embedding_rotate(self, scores): # -> Tensor:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...



class MCTCTLayerNorm(nn.Module):
    def __init__(self) -> None:
        ...

    def forward(self, hidden_states):
        ...



class MCTCTSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, input_tensor): # -> Any:
        ...



class MCTCTAttention(nn.Module):
    def __init__(self, config) -> None:
        ...

    def prune_heads(self, heads): # -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...



class MCTCTIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states):
        ...



class MCTCTOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, input_tensor): # -> Any:
        ...



class MCTCTLayer(nn.Module):
    def __init__(self, config: MCTCTConfig) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...

    def feed_forward_chunk(self, attention_output): # -> Any:
        ...



class MCTCTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MCTCTConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


MCTCT_START_DOCSTRING = ...
MCTCT_INPUTS_DOCSTRING = ...
class MCTCTEncoder(MCTCTPreTrainedModel):
    def __init__(self, config: MCTCTConfig) -> None:
        ...

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor, head_mask: torch.Tensor, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...) -> Union[Tuple, BaseModelOutput]:
        ...



@add_start_docstrings("The bare M-CTC-T Model transformer outputting raw hidden-states without any specific head on top.", MCTCT_START_DOCSTRING)
class MCTCTModel(MCTCTPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(MCTCT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality="audio", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...



@add_start_docstrings("""MCTCT Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""", MCTCT_START_DOCSTRING)
class MCTCTForCTC(MCTCTPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(MCTCT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC, expected_output=_CTC_EXPECTED_OUTPUT, expected_loss=_CTC_EXPECTED_LOSS)
    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.LongTensor] = ...) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...

