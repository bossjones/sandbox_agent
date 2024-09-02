"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_lilt import LiltConfig

"""PyTorch LiLT model."""
logger = ...
_CONFIG_FOR_DOC = ...
class LiltTextEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...): # -> tuple[Any, Any | Tensor]:
        ...

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Args:
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        ...

    def create_position_ids_from_inputs_embeds(self, inputs_embeds): # -> Tensor:
        """
        Args:
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        ...



class LiltLayoutEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, bbox=..., position_ids=...): # -> Any:
        ...



class LiltSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...

    def transpose_for_scores(self, x, r=...):
        ...

    def forward(self, hidden_states, layout_inputs, attention_mask=..., head_mask=..., output_attentions=...): # -> tuple[tuple[Tensor, Tensor], Any] | tuple[tuple[Tensor, Tensor]]:
        ...



class LiltSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class LiltAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...

    def prune_heads(self, heads): # -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, layout_inputs: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...



class LiltIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class LiltOutput(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...



class LiltLayer(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, layout_inputs: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...

    def feed_forward_chunk(self, attention_output): # -> Any:
        ...

    def layout_feed_forward_chunk(self, attention_output): # -> Any:
        ...



class LiltEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, layout_inputs: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        ...



class LiltPooler(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...



class LiltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LiltConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...


LILT_START_DOCSTRING = ...
LILT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare LiLT Model transformer outputting raw hidden-states without any specific head on top.", LILT_START_DOCSTRING)
class LiltModel(LiltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...

    def get_input_embeddings(self): # -> Embedding:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor] = ..., bbox: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        r"""

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
        >>> model = AutoModel.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = tokenizer(words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...



@add_start_docstrings("""
    LiLT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, LILT_START_DOCSTRING)
class LiltForSequenceClassification(LiltPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
        >>> model = AutoModelForSequenceClassification.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = tokenizer(words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> predicted_class_idx = outputs.logits.argmax(-1).item()
        >>> predicted_class = model.config.id2label[predicted_class_idx]
        ```"""
        ...



@add_start_docstrings("""
    Lilt Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, LILT_START_DOCSTRING)
class LiltForTokenClassification(LiltPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
        >>> model = AutoModelForTokenClassification.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = tokenizer(words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> predicted_class_indices = outputs.logits.argmax(-1)
        ```"""
        ...



class LiltClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config) -> None:
        ...

    def forward(self, features, **kwargs): # -> Any:
        ...



@add_start_docstrings("""
    Lilt Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, LILT_START_DOCSTRING)
class LiltForQuestionAnswering(LiltPreTrainedModel):
    def __init__(self, config) -> None:
        ...

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., bbox: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., start_positions: Optional[torch.LongTensor] = ..., end_positions: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        >>> from datasets import load_dataset

        >>> tokenizer = AutoTokenizer.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
        >>> model = AutoModelForQuestionAnswering.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = tokenizer(words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
        >>> predicted_answer = tokenizer.decode(predict_answer_tokens)
        ```"""
        ...

