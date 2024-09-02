"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_deberta import DebertaConfig

""" PyTorch DeBERTa model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_CHECKPOINT_FOR_MASKED_LM = ...
_MASKED_LM_EXPECTED_OUTPUT = ...
_MASKED_LM_EXPECTED_LOSS = ...
_CHECKPOINT_FOR_QA = ...
_QA_EXPECTED_OUTPUT = ...
_QA_EXPECTED_LOSS = ...
_QA_TARGET_START_INDEX = ...
_QA_TARGET_END_INDEX = ...
class ContextPooler(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states):
        ...

    @property
    def output_dim(self):
        ...



class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta.modeling_deberta import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""
    @staticmethod
    def forward(self, input, mask, dim): # -> Tensor:
        ...

    @staticmethod
    def backward(self, grad_output): # -> tuple[Tensor, None, None]:
        ...

    @staticmethod
    def symbolic(g, self, mask, dim): # -> Value | tuple[Value, ...]:
        ...



class DropoutContext:
    def __init__(self) -> None:
        ...



def get_mask(input, local_context): # -> tuple[Tensor | None, Any | int]:
    ...

class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""
    @staticmethod
    def forward(ctx, input, local_ctx):
        ...

    @staticmethod
    def backward(ctx, grad_output): # -> tuple[Any, None]:
        ...

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        ...



class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """
    def __init__(self, drop_prob) -> None:
        ...

    def forward(self, x): # -> Any | None:
        """
        Call the module

        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        ...

    def clear_context(self): # -> None:
        ...

    def init_context(self, reuse_mask=..., scale=...): # -> None:
        ...

    def get_context(self): # -> Any:
        ...



class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""
    def __init__(self, size, eps=...) -> None:
        ...

    def forward(self, hidden_states):
        ...



class DebertaSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, input_tensor): # -> Any:
        ...



class DebertaAttention(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask, output_attentions=..., query_states=..., relative_pos=..., rel_embeddings=...): # -> tuple[Any, Any] | Any:
        ...



class DebertaIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class DebertaOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, input_tensor): # -> Any:
        ...



class DebertaLayer(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask, query_states=..., relative_pos=..., rel_embeddings=..., output_attentions=...): # -> tuple[Any, Any] | Any:
        ...



class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""
    def __init__(self, config) -> None:
        ...

    def get_rel_embedding(self): # -> Tensor | None:
        ...

    def get_attention_mask(self, attention_mask):
        ...

    def get_rel_pos(self, hidden_states, query_states=..., relative_pos=...): # -> Tensor | None:
        ...

    def forward(self, hidden_states, attention_mask, output_hidden_states=..., output_attentions=..., query_states=..., relative_pos=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        ...



def build_relative_position(query_size, key_size, device): # -> Tensor:
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """
    ...

@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    ...

@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    ...

@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    ...

class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]

    """
    def __init__(self, config) -> None:
        ...

    def transpose_for_scores(self, x):
        ...

    def forward(self, hidden_states, attention_mask, output_attentions=..., query_states=..., relative_pos=..., rel_embeddings=...): # -> tuple[Tensor, Any] | Tensor:
        """
        Call the module

        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`torch.BoolTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        ...

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor): # -> Tensor | Literal[0]:
        ...



class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., mask=..., inputs_embeds=...): # -> Any:
        ...



class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DebertaConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_unexpected = ...
    supports_gradient_checkpointing = ...


DEBERTA_START_DOCSTRING = ...
DEBERTA_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.", DEBERTA_START_DOCSTRING)
class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def set_input_embeddings(self, new_embeddings): # -> None:
        ...

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...



@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
class DebertaForMaskedLM(DebertaPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...

    def get_output_embeddings(self): # -> Linear:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_MASKED_LM, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask="[MASK]", expected_output=_MASKED_LM_EXPECTED_OUTPUT, expected_loss=_MASKED_LM_EXPECTED_LOSS)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        ...



class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, sequence_output): # -> Any:
        ...



@add_start_docstrings("""
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, DEBERTA_START_DOCSTRING)
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def set_input_embeddings(self, new_embeddings): # -> None:
        ...

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...



@add_start_docstrings("""
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, DEBERTA_START_DOCSTRING)
class DebertaForTokenClassification(DebertaPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...



@add_start_docstrings("""
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, DEBERTA_START_DOCSTRING)
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_QA, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC, expected_output=_QA_EXPECTED_OUTPUT, expected_loss=_QA_EXPECTED_LOSS, qa_target_start_index=_QA_TARGET_START_INDEX, qa_target_end_index=_QA_TARGET_END_INDEX)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., start_positions: Optional[torch.Tensor] = ..., end_positions: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...

