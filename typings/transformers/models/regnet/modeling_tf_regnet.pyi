"""
This type stub file was generated by pyright.
"""

import tensorflow as tf
from typing import Optional, Tuple, Union
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import TFBaseModelOutputWithNoAttention, TFBaseModelOutputWithPoolingAndNoAttention, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras, keras_serializable, unpack_inputs
from .configuration_regnet import RegNetConfig

""" TensorFlow RegNet model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
class TFRegNetConvLayer(keras.layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = ..., stride: int = ..., groups: int = ..., activation: Optional[str] = ..., **kwargs) -> None:
        ...

    def call(self, hidden_state):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetEmbeddings(keras.layers.Layer):
    """
    RegNet Embeddings (stem) composed of a single aggressive convolution.
    """
    def __init__(self, config: RegNetConfig, **kwargs) -> None:
        ...

    def call(self, pixel_values):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetShortCut(keras.layers.Layer):
    """
    RegNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = ..., **kwargs) -> None:
        ...

    def call(self, inputs: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetSELayer(keras.layers.Layer):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """
    def __init__(self, in_channels: int, reduced_channels: int, **kwargs) -> None:
        ...

    def call(self, hidden_state):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetXLayer(keras.layers.Layer):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = ..., **kwargs) -> None:
        ...

    def call(self, hidden_state):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetYLayer(keras.layers.Layer):
    """
    RegNet's Y layer: an X layer with Squeeze and Excitation.
    """
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = ..., **kwargs) -> None:
        ...

    def call(self, hidden_state):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetStage(keras.layers.Layer):
    """
    A RegNet stage composed by stacked layers.
    """
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = ..., depth: int = ..., **kwargs) -> None:
        ...

    def call(self, hidden_state):
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetEncoder(keras.layers.Layer):
    def __init__(self, config: RegNetConfig, **kwargs) -> None:
        ...

    def call(self, hidden_state: tf.Tensor, output_hidden_states: bool = ..., return_dict: bool = ...) -> TFBaseModelOutputWithNoAttention:
        ...

    def build(self, input_shape=...): # -> None:
        ...



@keras_serializable
class TFRegNetMainLayer(keras.layers.Layer):
    config_class = RegNetConfig
    def __init__(self, config, **kwargs) -> None:
        ...

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> TFBaseModelOutputWithPoolingAndNoAttention:
        ...

    def build(self, input_shape=...): # -> None:
        ...



class TFRegNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RegNetConfig
    base_model_prefix = ...
    main_input_name = ...
    @property
    def input_signature(self): # -> dict[str, Any]:
        ...



REGNET_START_DOCSTRING = ...
REGNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare RegNet model outputting raw features without any specific head on top.", REGNET_START_DOCSTRING)
class TFRegNetModel(TFRegNetPreTrainedModel):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: tf.Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        ...

    def build(self, input_shape=...): # -> None:
        ...



@add_start_docstrings("""
    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """, REGNET_START_DOCSTRING)
class TFRegNetForImageClassification(TFRegNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: RegNetConfig, *inputs, **kwargs) -> None:
        ...

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: Optional[tf.Tensor] = ..., labels: Optional[tf.Tensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

    def build(self, input_shape=...): # -> None:
        ...

