"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, keras, keras_serializable, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_convbert import ConvBertConfig

""" TF 2.0 ConvBERT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
class TFConvBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: ConvBertConfig, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, input_ids: tf.Tensor = ..., position_ids: tf.Tensor = ..., token_type_ids: tf.Tensor = ..., inputs_embeds: tf.Tensor = ..., past_key_values_length=..., training: bool = ...) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...



class TFConvBertSelfAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def transpose_for_scores(self, x, batch_size):
        ...

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=...): # -> tuple[Any, Any] | tuple[Any]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertSelfOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states, input_tensor, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertAttention(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def prune_heads(self, heads):
        ...

    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class GroupedLinearLayer(keras.layers.Layer):
    def __init__(self, input_size, output_size, num_groups, kernel_initializer, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, hidden_states):
        ...



class TFConvBertIntermediate(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertOutput(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states, input_tensor, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertEncoder(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=...): # -> tuple[Any, ...] | TFBaseModelOutput:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFConvBertMainLayer(keras.layers.Layer):
    config_class = ConvBertConfig
    def __init__(self, config, **kwargs) -> None:
        ...

    def get_input_embeddings(self): # -> TFConvBertEmbeddings:
        ...

    def set_input_embeddings(self, value): # -> None:
        ...

    def get_extended_attention_mask(self, attention_mask, input_shape, dtype):
        ...

    def get_head_mask(self, head_mask):
        ...

    @unpack_inputs
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ConvBertConfig
    base_model_prefix = ...


CONVBERT_START_DOCSTRING = ...
CONVBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare ConvBERT Model transformer outputting raw hidden-states without any specific head on top.", CONVBERT_START_DOCSTRING)
class TFConvBertModel(TFConvBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: Optional[Union[np.array, tf.Tensor]] = ..., token_type_ids: Optional[Union[np.array, tf.Tensor]] = ..., position_ids: Optional[Union[np.array, tf.Tensor]] = ..., head_mask: Optional[Union[np.array, tf.Tensor]] = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertMaskedLMHead(keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...

    def build(self, input_shape): # -> None:
        ...

    def get_output_embeddings(self): # -> Any:
        ...

    def set_output_embeddings(self, value): # -> None:
        ...

    def get_bias(self): # -> dict[str, Any]:
        ...

    def set_bias(self, value): # -> None:
        ...

    def call(self, hidden_states):
        ...



class TFConvBertGeneratorPredictions(keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, generator_hidden_states, training=...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""ConvBERT Model with a `language modeling` head on top.""", CONVBERT_START_DOCSTRING)
class TFConvBertForMaskedLM(TFConvBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    def get_lm_head(self): # -> TFConvBertMaskedLMHead:
        ...

    def get_prefix_bias_name(self):
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[Tuple, TFMaskedLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFConvBertClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states, **kwargs):
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    ConvBERT Model transformer with a sequence classification/regression head on top e.g., for GLUE tasks.
    """, CONVBERT_START_DOCSTRING)
class TFConvBertForSequenceClassification(TFConvBertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[Tuple, TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    ConvBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, CONVBERT_START_DOCSTRING)
class TFConvBertForMultipleChoice(TFConvBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[Tuple, TFMultipleChoiceModelOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, CONVBERT_START_DOCSTRING)
class TFConvBertForTokenClassification(TFConvBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[Tuple, TFTokenClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    ConvBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, CONVBERT_START_DOCSTRING)
class TFConvBertForQuestionAnswering(TFConvBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., head_mask: np.ndarray | tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: tf.Tensor | None = ..., end_positions: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[Tuple, TFQuestionAnsweringModelOutput]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...

