"""
This type stub file was generated by pyright.
"""

import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFPreTrainedModel, keras, keras_serializable, unpack_inputs
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING

""" TensorFlow BLIP model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    ...

def blip_loss(similarity: tf.Tensor) -> tf.Tensor:
    ...

@dataclass
class TFBlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`tf.Tensor`, *optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.`
    """
    loss: Tuple[tf.Tensor] | None = ...
    logits: Tuple[tf.Tensor] | None = ...
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor = ...
    hidden_states: Tuple[tf.Tensor, ...] | None = ...
    attentions: Tuple[tf.Tensor, ...] | None = ...
    @property
    def decoder_logits(self): # -> Tuple[Any] | None:
        ...



@dataclass
class TFBlipTextVisionModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: tf.Tensor | None = ...
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor = ...
    hidden_states: Tuple[tf.Tensor, ...] | None = ...
    attentions: Tuple[tf.Tensor, ...] | None = ...


@dataclass
class TFBlipImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        itm_score (`tf.Tensor`):
            The image-text similarity scores.
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        vision_pooler_output (`tf.Tensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the vision of the vision-only branch of the model.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        question_embeds (`tf.Tensor`):
            The question embeddings obtained by the text projection layer.
    """
    itm_score: tf.Tensor | None = ...
    loss: tf.Tensor | None = ...
    image_embeds: tf.Tensor | None = ...
    last_hidden_state: tf.Tensor = ...
    hidden_states: Tuple[tf.Tensor, ...] | None = ...
    vision_pooler_output: tf.Tensor | None = ...
    attentions: Tuple[tf.Tensor, ...] | None = ...
    question_embeds: Tuple[tf.Tensor] | None = ...


@dataclass
class TFBlipOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
    """
    loss: tf.Tensor | None = ...
    logits_per_image: tf.Tensor = ...
    logits_per_text: tf.Tensor = ...
    text_embeds: tf.Tensor = ...
    image_embeds: tf.Tensor = ...
    text_model_output: TFBaseModelOutputWithPooling = ...
    vision_model_output: TFBaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...



class TFBlipVisionEmbeddings(keras.layers.Layer):
    def __init__(self, config: BlipVisionConfig, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        ...



class TFBlipTextEmbeddings(keras.layers.Layer):
    def __init__(self, config: BlipTextConfig, **kwargs) -> None:
        ...

    def build(self, input_shape: tf.TensorShape = ...): # -> None:
        ...

    def call(self, input_ids: tf.Tensor = ..., position_ids: tf.Tensor = ..., inputs_embeds: tf.Tensor = ...) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...



class TFBlipAttention(keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., training: Optional[bool] = ...) -> Tuple[tf.Tensor, tf.Tensor | None, Tuple[tf.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBlipMLP(keras.layers.Layer):
    def __init__(self, config: BlipConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBlipEncoderLayer(keras.layers.Layer):
    def __init__(self, config: BlipConfig, **kwargs) -> None:
        ...

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, output_attentions: Optional[bool] = ..., training: Optional[bool] = ...) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBlipPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BlipConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


BLIP_START_DOCSTRING = ...
BLIP_VISION_INPUTS_DOCSTRING = ...
BLIP_INPUTS_DOCSTRING = ...
@keras_serializable
class TFBlipEncoder(keras.layers.Layer):
    config_class = BlipConfig
    def __init__(self, config: BlipConfig, **kwargs) -> None:
        ...

    @unpack_inputs
    def call(self, inputs_embeds, attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBlipVisionModel(TFBlipPreTrainedModel):
    main_input_name = ...
    config_class = BlipVisionConfig
    def __init__(self, config: BlipVisionConfig, *args, **kwargs) -> None:
        ...

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def call(self, pixel_values: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        ...

    def get_input_embeddings(self): # -> TFBlipVisionEmbeddings:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFBlipMainLayer(keras.layers.Layer):
    config_class = BlipConfig
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None:
        ...

    def build(self, input_shape=...): # -> None:
        ...

    @unpack_inputs
    def call(self, input_ids: tf.Tensor | None = ..., pixel_values: tf.Tensor | None = ..., attention_mask: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBlipOutput]:
        ...



class TFBlipModel(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    def __init__(self, config: BlipConfig, *inputs, **kwargs) -> None:
        ...

    def serving_output(self, output: TFBlipOutput) -> TFBlipOutput:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipOutput, config_class=BlipConfig)
    def call(self, input_ids: tf.Tensor | None = ..., pixel_values: tf.Tensor | None = ..., attention_mask: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBlipOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""
        ...

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: tf.Tensor | None = ..., attention_mask: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., return_dict: Optional[bool] = ...) -> tf.Tensor:
        r"""
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFBlipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: tf.Tensor | None = ..., return_dict: Optional[bool] = ...) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFBlipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.
    """, BLIP_START_DOCSTRING)
class TFBlipForConditionalGeneration(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ...
    main_input_name = ...
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None:
        ...

    def get_input_embeddings(self) -> keras.layers.Layer:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipForConditionalGenerationModelOutput, config_class=BlipConfig)
    def call(self, pixel_values: tf.Tensor, input_ids: tf.Tensor | None = ..., attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., labels: tf.Tensor | None = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBlipForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model(**inputs)
        ```"""
        ...

    def generate(self, pixel_values: tf.Tensor, input_ids: tf.Tensor | None = ..., attention_mask: tf.Tensor | None = ..., **generate_kwargs) -> tf.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        two cats sleeping on a couch
        ```
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
    with the encoding of the image, and the text decoder will output the answer to the question.
    """, BLIP_START_DOCSTRING)
class TFBlipForQuestionAnswering(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None:
        ...

    def get_input_embeddings(self) -> keras.layers.Layer:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def call(self, input_ids: tf.Tensor, pixel_values: tf.Tensor | None = ..., decoder_input_ids: tf.Tensor | None = ..., decoder_attention_mask: tf.Tensor | None = ..., attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., labels: tf.Tensor | None = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBlipTextVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # training
        >>> text = "How many cats are in the picture?"
        >>> label = "2"
        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> labels = processor(text=label, return_tensors="tf").input_ids

        >>> inputs["labels"] = labels
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss

        >>> # inference
        >>> text = "How many cats are in the picture?"
        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```"""
        ...

    def generate(self, input_ids: tf.Tensor, pixel_values: tf.Tensor, attention_mask: tf.Tensor | None = ..., **generate_kwargs) -> tf.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            generate_kwargs (dict, *optional*):
                Additional arguments passed to the `generate` function of the decoder


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of
    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
    the image.
    """, BLIP_START_DOCSTRING)
class TFBlipForImageTextRetrieval(TFBlipPreTrainedModel):
    config_class = BlipConfig
    def __init__(self, config: BlipConfig, *args, **kwargs) -> None:
        ...

    def get_input_embeddings(self) -> keras.layers.Layer:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipImageTextMatchingModelOutput, config_class=BlipVisionConfig)
    def call(self, input_ids: tf.Tensor, pixel_values: tf.Tensor | None = ..., use_itm_head: Optional[bool] = ..., attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[Tuple, TFBlipImageTextMatchingModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForImageTextRetrieval

        >>> model = TFBlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> outputs = model(**inputs)
        ```
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...

