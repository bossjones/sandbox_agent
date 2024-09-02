"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from torch.autograd.function import Function
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_reformer import ReformerConfig

"""PyTorch REFORMER model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LSHSelfAttentionOutput = ...
LocalSelfAttentionOutput = ...
AttentionOutput = ...
ReformerOutput = ...
ReformerBackwardOutput = ...
ReformerEncoderOutput = ...
class AxialPositionEmbeddings(nn.Module):
    """
    Constructs axial position embeddings. Useful for very long input sequences to save memory and time.
    """
    def __init__(self, config) -> None:
        ...

    def forward(self, position_ids): # -> Tensor:
        ...



class PositionEmbeddings(nn.Module):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`."""
    def __init__(self, config) -> None:
        ...

    def forward(self, position_ids): # -> Tensor:
        ...



class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...

    def forward(self, input_ids=..., position_ids=..., inputs_embeds=..., start_idx_pos_encodings=...): # -> Any:
        ...



class EfficientAttentionMixin:
    """
    A few utilities for nn.Modules in Reformer, to be used as a mixin.
    """
    ...


class LSHSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., buckets=..., past_buckets_states=..., use_cache=..., output_attentions=..., **kwargs):
        ...



class ReverseSort(Function):
    """
    After chunked attention is applied which sorted clusters, original ordering has to be restored. Since customized
    backward function is used for Reformer, the gradients of the output vectors have to be explicitly sorted here.
    """
    @staticmethod
    def forward(ctx, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx): # -> tuple[Tensor, Tensor]:
        ...

    @staticmethod
    def backward(ctx, grad_out_vectors, grad_logits): # -> tuple[Tensor, Tensor, None, None]:
        ...



class LocalSelfAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., past_buckets_states=..., use_cache=..., output_attentions=..., **kwargs): # -> LocalSelfAttentionOutput:
        ...



class ReformerSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Tensor:
        ...



class ReformerAttention(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., past_buckets_states=..., use_cache=..., orig_sequence_length=..., output_attentions=..., buckets=...): # -> AttentionOutput:
        ...



class ReformerFeedForwardDense(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states):
        ...



class ReformerFeedForwardOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Tensor:
        ...



class ChunkReformerFeedForward(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, attention_output): # -> Tensor:
        ...

    def forward_chunk(self, hidden_states): # -> Any:
        ...



class ReformerLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, prev_attn_output, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., past_buckets_states=..., use_cache=..., orig_sequence_length=..., output_attentions=...): # -> ReformerOutput:
        ...

    def backward_pass(self, next_attn_output, hidden_states, grad_attn_output, grad_hidden_states, attention_mask=..., head_mask=..., buckets=...): # -> ReformerBackwardOutput:
        ...



class _ReversibleFunction(Function):
    """
    To prevent PyTorch from performing the usual backpropagation, a customized backward function is implemented here.
    This way it is made sure that no memory expensive activations are saved during the forward pass. This function is
    heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    """
    @staticmethod
    def forward(ctx, hidden_states, layers, attention_mask, head_mask, num_hashes, all_hidden_states, all_attentions, past_buckets_states, use_cache, orig_sequence_length, output_hidden_states, output_attentions): # -> Tensor:
        ...

    @staticmethod
    def backward(ctx, grad_hidden_states): # -> tuple[Tensor, None, None, None, None, None, None, None, None, None, None, None]:
        ...



class ReformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, attention_mask=..., head_mask=..., num_hashes=..., past_buckets_states=..., use_cache=..., orig_sequence_length=..., output_hidden_states=..., output_attentions=...): # -> ReformerEncoderOutput:
        ...



class ReformerOnlyLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states): # -> Tensor:
        ...

    def forward_chunk(self, hidden_states): # -> Any:
        ...



class ReformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ReformerConfig
    base_model_prefix = ...
    @property
    def dummy_inputs(self): # -> dict[str, Tensor]:
        ...



@dataclass
class ReformerModelOutput(ModelOutput):
    """
    Output type of [`ReformerModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        past_buckets_states (`List[Tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `Tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

            Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
            up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor
    past_buckets_states: Optional[List[Tuple[torch.LongTensor, torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class ReformerModelWithLMHeadOutput(ModelOutput):
    """
    Output type of [`ReformerModelWithLMHead`].

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        past_buckets_states (`List[Tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `Tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

            Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
            up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            TTuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    past_buckets_states: Optional[List[Tuple[torch.LongTensor, torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


REFORMER_START_DOCSTRING = ...
REFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Reformer Model transformer outputting raw hidden-stateswithout any specific head on top.", REFORMER_START_DOCSTRING)
class ReformerModel(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=ReformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., num_hashes: Optional[int] = ..., past_buckets_states: Optional[List[Tuple[torch.Tensor]]] = ..., use_cache: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., output_attentions: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ReformerModelOutput]:
        ...



@add_start_docstrings("""Reformer Model with a `language modeling` head on top.""", REFORMER_START_DOCSTRING)
class ReformerModelWithLMHead(ReformerPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...

    def get_output_embeddings(self): # -> Linear:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., num_hashes: Optional[int] = ..., past_buckets_states: Optional[List[Tuple[torch.Tensor]]] = ..., use_cache: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., output_attentions: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: Optional[torch.Tensor] = ...) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
                config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
                labels in `[0, ..., config.vocab_size]`
        """
        ...

    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., use_cache=..., num_hashes=..., **kwargs): # -> dict[str, Any]:
        ...



@add_start_docstrings("""Reformer Model with a `language modeling` head on top.""", REFORMER_START_DOCSTRING)
class ReformerForMaskedLM(ReformerPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...

    def get_output_embeddings(self): # -> Linear:
        ...

    def set_output_embeddings(self, new_embeddings): # -> None:
        ...

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., num_hashes: Optional[int] = ..., labels: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., output_attentions: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels

        Returns:

        <Tip warning={true}>

        This example uses a false checkpoint since we don't have any available pretrained model for the masked language
        modeling task with the Reformer architecture.

        </Tip>

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, ReformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-reformer")
        >>> model = ReformerForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-reformer")

        >>> # add mask_token
        >>> tokenizer.add_special_tokens({"mask_token": "[MASK]"})  # doctest: +IGNORE_RESULT
        >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

        >>> # resize model's embedding matrix
        >>> model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + 1)  # doctest: +IGNORE_RESULT

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # retrieve index of [MASK]
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> predicted_token = tokenizer.decode(predicted_token_id)
        ```

        ```python
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
        >>> # mask labels of non-[MASK] tokens
        >>> labels = torch.where(
        ...     inputs.input_ids == tokenizer.mask_token_id, labels[:, : inputs["input_ids"].shape[-1]], -100
        ... )

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = round(outputs.loss.item(), 2)
        ```
        """
        ...



@add_start_docstrings("""
    Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, REFORMER_START_DOCSTRING)
class ReformerForSequenceClassification(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., num_hashes: Optional[int] = ..., labels: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., output_attentions: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example of single-label classification:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, ReformerForSequenceClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
        >>> model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> predicted_class_id = logits.argmax().item()
        >>> label = model.config.id2label[predicted_class_id]
        ```

        ```python
        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
        >>> num_labels = len(model.config.id2label)
        >>> model = ReformerForSequenceClassification.from_pretrained(
        ...     "google/reformer-crime-and-punishment", num_labels=num_labels
        ... )

        >>> labels = torch.tensor(1)
        >>> loss = model(**inputs, labels=labels).loss
        ```
        """
        ...



class ReformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states, **kwargs): # -> Any:
        ...



@add_start_docstrings("""
    Reformer Model with a span classification head on top for extractive question-answering tasks like SQuAD / TriviaQA
    ( a linear layer on top of hidden-states output to compute `span start logits` and `span end logits`.
    """, REFORMER_START_DOCSTRING)
class ReformerForQuestionAnswering(ReformerPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., num_hashes: Optional[int] = ..., start_positions: Optional[torch.Tensor] = ..., end_positions: Optional[torch.Tensor] = ..., output_hidden_states: Optional[bool] = ..., output_attentions: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, QuestionAnsweringModelOutput]:
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

