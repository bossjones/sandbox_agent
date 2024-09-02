"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFModelInputType, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, keras, unpack_inputs
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_deberta_v2 import DebertaV2Config

""" TF 2.0 DeBERTa-v2 model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
class TFDebertaV2ContextPooler(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def call(self, hidden_states, training: bool = ...):
        ...

    @property
    def output_dim(self) -> int:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2XSoftmax(keras.layers.Layer):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`tf.Tensor`): The input tensor that will apply softmax.
        mask (`tf.Tensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax
    """
    def __init__(self, axis=..., **kwargs) -> None:
        ...

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):
        ...



class TFDebertaV2StableDropout(keras.layers.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """
    def __init__(self, drop_prob, **kwargs) -> None:
        ...

    @tf.custom_gradient
    def xdropout(self, inputs): # -> tuple[Any, Callable[..., Any]]:
        """
        Applies dropout to the inputs, as vanilla dropout, but also scales the remaining elements up by 1/drop_prob.
        """
        ...

    def call(self, inputs: tf.Tensor, training: tf.Tensor = ...): # -> tuple[Any, Callable[..., Any]]:
        ...



class TFDebertaV2SelfOutput(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def call(self, hidden_states, input_tensor, training: bool = ...):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2Attention(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor = ..., relative_pos: tf.Tensor = ..., rel_embeddings: tf.Tensor = ..., output_attentions: bool = ..., training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2Intermediate(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2Output(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2Layer(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor = ..., relative_pos: tf.Tensor = ..., rel_embeddings: tf.Tensor = ..., output_attentions: bool = ..., training: bool = ...) -> Tuple[tf.Tensor]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2ConvLayer(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, hidden_states: tf.Tensor, residual_states: tf.Tensor, input_mask: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...



class TFDebertaV2Encoder(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def get_rel_embedding(self): # -> None:
        ...

    def get_attention_mask(self, attention_mask):
        ...

    def get_rel_pos(self, hidden_states, query_states=..., relative_pos=...): # -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor = ..., relative_pos: tf.Tensor = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...



def make_log_bucket_position(relative_pos, bucket_size, max_position):
    ...

def build_relative_position(query_size, key_size, bucket_size=..., max_position=...):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        `tf.Tensor`: A tensor with shape [1, query_size, key_size]

    """
    ...

def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    ...

def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    ...

def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    ...

def take_along_axis(x, indices):
    ...

class TFDebertaV2DisentangledSelfAttention(keras.layers.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def transpose_for_scores(self, tensor: tf.Tensor, attention_heads: int) -> tf.Tensor:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor = ..., relative_pos: tf.Tensor = ..., rel_embeddings: tf.Tensor = ..., output_attentions: bool = ..., training: bool = ...) -> Tuple[tf.Tensor]:
        """
        Call the module

        Args:
            hidden_states (`tf.Tensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`tf.Tensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            return_att (`bool`, optional):
                Whether return the attention matrix.

            query_states (`tf.Tensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`tf.Tensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`tf.Tensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        ...

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor): # -> Literal[0]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2Embeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, input_ids: tf.Tensor = ..., position_ids: tf.Tensor = ..., token_type_ids: tf.Tensor = ..., inputs_embeds: tf.Tensor = ..., mask: tf.Tensor = ..., training: bool = ...) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...



class TFDebertaV2PredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2LMPredictionHead(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, input_embeddings: keras.layers.Layer, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def get_output_embeddings(self) -> keras.layers.Layer:
        ...

    def set_output_embeddings(self, value: tf.Variable): # -> None:
        ...

    def get_bias(self) -> Dict[str, tf.Variable]:
        ...

    def set_bias(self, value: tf.Variable): # -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...



class TFDebertaV2OnlyMLMHead(keras.layers.Layer):
    def __init__(self, config: DebertaV2Config, input_embeddings: keras.layers.Layer, **kwargs) -> None:
        ...

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2MainLayer(keras.layers.Layer):
    config_class = DebertaV2Config
    def __init__(self, config: DebertaV2Config, **kwargs) -> None:
        ...

    def get_input_embeddings(self) -> keras.layers.Layer:
        ...

    def set_input_embeddings(self, value: tf.Variable): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFDebertaV2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DebertaV2Config
    base_model_prefix = ...


DEBERTA_START_DOCSTRING = ...
DEBERTA_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.", DEBERTA_START_DOCSTRING)
class TFDebertaV2Model(TFDebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
class TFDebertaV2ForMaskedLM(TFDebertaV2PreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None:
        ...

    def get_lm_head(self) -> keras.layers.Layer:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, DEBERTA_START_DOCSTRING)
class TFDebertaV2ForSequenceClassification(TFDebertaV2PreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, DEBERTA_START_DOCSTRING)
class TFDebertaV2ForTokenClassification(TFDebertaV2PreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, DEBERTA_START_DOCSTRING)
class TFDebertaV2ForQuestionAnswering(TFDebertaV2PreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., start_positions: np.ndarray | tf.Tensor | None = ..., end_positions: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    DeBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, DEBERTA_START_DOCSTRING)
class TFDebertaV2ForMultipleChoice(TFDebertaV2PreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config: DebertaV2Config, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None = ..., attention_mask: np.ndarray | tf.Tensor | None = ..., token_type_ids: np.ndarray | tf.Tensor | None = ..., position_ids: np.ndarray | tf.Tensor | None = ..., inputs_embeds: np.ndarray | tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., labels: np.ndarray | tf.Tensor | None = ..., training: Optional[bool] = ...) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...

