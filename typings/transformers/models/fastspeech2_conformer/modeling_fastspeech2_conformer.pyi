"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, replace_return_docstrings
from .configuration_fastspeech2_conformer import FastSpeech2ConformerConfig, FastSpeech2ConformerHifiGanConfig, FastSpeech2ConformerWithHifiGanConfig

""" PyTorch FastSpeech2Conformer model."""
logger = ...
@dataclass
class FastSpeech2ConformerModelOutput(ModelOutput):
    """
    Output type of [`FastSpeech2ConformerModel`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Spectrogram generation loss.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The predicted spectrogram.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        duration_outputs (`torch.LongTensor` of shape `(batch_size, max_text_length + 1)`, *optional*):
            Outputs of the duration predictor.
        pitch_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the pitch predictor.
        energy_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the energy predictor.

    """
    loss: Optional[torch.FloatTensor] = ...
    spectrogram: torch.FloatTensor = ...
    encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    duration_outputs: torch.LongTensor = ...
    pitch_outputs: torch.FloatTensor = ...
    energy_outputs: torch.FloatTensor = ...


@dataclass
class FastSpeech2ConformerWithHifiGanOutput(FastSpeech2ConformerModelOutput):
    """
    Output type of [`FastSpeech2ConformerWithHifiGan`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
            Speech output as a result of passing the predicted mel spectrogram through the vocoder.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Spectrogram generation loss.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The predicted spectrogram.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        duration_outputs (`torch.LongTensor` of shape `(batch_size, max_text_length + 1)`, *optional*):
            Outputs of the duration predictor.
        pitch_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the pitch predictor.
        energy_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the energy predictor.
    """
    waveform: torch.FloatTensor = ...


_CONFIG_FOR_DOC = ...
FASTSPEECH2_CONFORMER_START_DOCSTRING = ...
HIFIGAN_START_DOCSTRING = ...
FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING = ...
def length_regulator(encoded_embeddings, duration_labels, speaking_speed=...):
    """
    Length regulator for feed-forward Transformer.

    This is the length regulator module described in `FastSpeech: Fast, Robust and Controllable Text to Speech`
    https://arxiv.org/pdf/1905.09263.pdf. The length regulator expands char or phoneme-level embedding features to
    frame-level by repeating each feature based on the corresponding predicted durations.

    Args:
        encoded_embeddings (`torch.Tensor` of shape `(batch_size, max_text_length, embedding_dim)`):
            Batch of sequences of char or phoneme embeddings.
        duration_labels (`torch.LongTensor` of shape `(batch_size, time)`):
            Batch of durations of each frame.
        speaking_speed (`float`, *optional*, defaults to 1.0):
            Value to control speed of speech.

    Returns:
        `torch.Tensor`:
            Replicated input tensor based on durations (batch_size, time*, embedding_dim).
    """
    ...

class FastSpeech2ConformerDurationPredictor(nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described in the paper 'FastSpeech: Fast, Robust and Controllable Text to
    Speech' https://arxiv.org/pdf/1905.09263.pdf The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`, the
        outputs are calculated in log domain but in `inference`, those are calculated in linear domain.

    """
    def __init__(self, config: FastSpeech2ConformerConfig) -> None:
        ...

    def forward(self, encoder_hidden_states): # -> Tensor | Any:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            `torch.Tensor`: Batch of predicted durations in log domain `(batch_size, max_text_length)`.

        """
        ...



class FastSpeech2ConformerBatchNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=...) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):
    def __init__(self, config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor): # -> tuple[Any, Any]:
        ...



class FastSpeech2ConformerPredictorLayer(nn.Module):
    def __init__(self, input_channels, num_chans, kernel_size, dropout_rate) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class FastSpeech2ConformerVariancePredictor(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, num_layers=..., num_chans=..., kernel_size=..., dropout_rate=...) -> None:
        """
        Initilize variance predictor module.

        Args:
            input_dim (`int`): Input dimension.
            num_layers (`int`, *optional*, defaults to 2): Number of convolutional layers.
            num_chans (`int`, *optional*, defaults to 384): Number of channels of convolutional layers.
            kernel_size (`int`, *optional*, defaults to 3): Kernel size of convolutional layers.
            dropout_rate (`float`, *optional*, defaults to 0.5): Dropout rate.
        """
        ...

    def forward(self, encoder_hidden_states, padding_masks=...): # -> Any:
        """
        Calculate forward propagation.

        Args:
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            Tensor: Batch of predicted sequences `(batch_size, max_text_length, 1)`.
        """
        ...



class FastSpeech2ConformerVarianceEmbedding(nn.Module):
    def __init__(self, in_channels=..., out_channels=..., kernel_size=..., padding=..., dropout_rate=...) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        ...



class FastSpeech2ConformerAttention(nn.Module):
    """
    Multi-Head attention layer with relative position encoding. Details can be found in
    https://github.com/espnet/espnet/pull/2816. Paper: https://arxiv.org/abs/1901.02860.
    """
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None:
        """Construct an FastSpeech2ConformerAttention object."""
        ...

    def shift_relative_position_tensor(self, pos_tensor): # -> Tensor:
        """
        Args:
            pos_tensor (torch.Tensor of shape (batch_size, head, time1, 2*time1-1)): Input tensor.
        """
        ...

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., pos_emb: Optional[torch.Tensor] = ..., output_attentions: Optional[torch.Tensor] = ...) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time2, size)`): Values of the hidden states
            attention_mask (`torch.Tensor` of shape `(batch, time1, time2)`): Mask tensor.
            pos_emb (`torch.Tensor` of shape `(batch, 2*time1-1, size)`): Positional embedding tensor.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time1, d_model)`.
        """
        ...



class FastSpeech2ConformerConvolutionModule(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None:
        ...

    def forward(self, hidden_states): # -> Any:
        """
        Compute convolution module.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, channels)`): Input tensor.

        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, channels)`.

        """
        ...



class FastSpeech2ConformerEncoderLayer(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None:
        ...

    def forward(self, hidden_states: torch.Tensor, pos_emb: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[torch.Tensor] = ...): # -> tuple[Tensor, Any] | tuple[Tensor]:
        """
        Compute encoded features.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, size)`): Input tensor.
            pos_emb (`torch.Tensor` of shape `(1, time, size)`): Positional embeddings tensor.
            attention_mask (`torch.Tensor` of shape `(batch, time)`): Attention mask tensor for the input.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, size)`.

        """
        ...



class FastSpeech2ConformerMultiLayeredConv1d(nn.Module):
    """
    Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed to replace positionwise feed-forward network in Transformer
    block, which is introduced in 'FastSpeech: Fast, Robust and Controllable Text to Speech'
    https://arxiv.org/pdf/1905.09263.pdf
    """
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None:
        """
        Initialize FastSpeech2ConformerMultiLayeredConv1d module.

        Args:
            input_channels (`int`): Number of input channels.
            hidden_channels (`int`): Number of hidden channels.
            kernel_size (`int`): Kernel size of conv1d.
            dropout_rate (`float`): Dropout rate.
        """
        ...

    def forward(self, hidden_states): # -> Any:
        """
        Calculate forward propagation.

        Args:
            hidden_states (torch.Tensor): Batch of input tensors (batch_size, time, input_channels).

        Returns:
            torch.Tensor: Batch of output tensors (batch_size, time, hidden_channels).
        """
        ...



class FastSpeech2ConformerRelPositionalEncoding(nn.Module):
    """
    Args:
    Relative positional encoding module (new implementation). Details can be found in
    https://github.com/espnet/espnet/pull/2816. See : Appendix Batch in https://arxiv.org/abs/1901.02860
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance.
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
    """
    def __init__(self, config: FastSpeech2ConformerConfig, module_config) -> None:
        """
        Construct an PositionalEncoding object.
        """
        ...

    def extend_pos_enc(self, x): # -> None:
        """Reset the positional encodings."""
        ...

    def forward(self, feature_representation): # -> tuple[Any, Any]:
        """
        Args:
            feature_representation (`torch.Tensor` of shape (batch_size, time, `*`)):
                Input tensor.

        Returns:
            `torch.Tensor`: Encoded tensor (batch_size, time, `*`).
        """
        ...



class FastSpeech2ConformerEncoder(nn.Module):
    """
    FastSpeech2ConformerEncoder encoder module.

    Args:
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance.
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
        use_encoder_input_layer (`bool`, *optional*, defaults to `False`):
            Input layer type.
    """
    def __init__(self, config: FastSpeech2ConformerConfig, module_config, use_encoder_input_layer=...) -> None:
        ...

    def forward(self, input_tensor: torch.LongTensor, attention_mask: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., output_attentions: Optional[bool] = ..., return_dict: Optional[bool] = ...): # -> tuple[Any, ...] | BaseModelOutput:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
            `torch.Tensor`:
                Output tensor of shape `(batch, time, attention_dim)`.
        """
        ...



class FastSpeech2ConformerLoss(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig) -> None:
        ...

    def forward(self, outputs_after_postnet, outputs_before_postnet, duration_outputs, pitch_outputs, energy_outputs, spectrogram_labels, duration_labels, pitch_labels, energy_labels, duration_mask, spectrogram_mask): # -> Any:
        """
        Args:
            outputs_after_postnet (`torch.Tensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of outputs after postnet.
            outputs_before_postnet (`torch.Tensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of outputs before postnet.
            duration_outputs (`torch.LongTensor` of shape `(batch_size, max_text_length)`):
                Batch of outputs of duration predictor.
            pitch_outputs (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of outputs of pitch predictor.
            energy_outputs (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of outputs of energy predictor.
            spectrogram_labels (`torch.Tensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, max_text_length)`): Batch of durations.
            pitch_labels (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of target token-averaged pitch.
            energy_labels (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of target token-averaged energy.
            duration_mask (`torch.LongTensor`):
                Mask used to discern which values the duration loss should be calculated for.
            spectrogram_mask (`torch.LongTensor`):
                Mask used to discern which values the spectrogam loss should be calculated for.

        Returns:
            `tuple(torch.FloatTensor)`: Tuple of tensors containing, in order, the L1 loss value, duration predictor
            loss value, pitch predictor loss value, and energy predictor loss value.

        """
        ...



class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = FastSpeech2ConformerConfig
    base_model_prefix = ...
    main_input_name = ...


@add_start_docstrings("""FastSpeech2Conformer Model.""", FASTSPEECH2_CONFORMER_START_DOCSTRING)
class FastSpeech2ConformerModel(FastSpeech2ConformerPreTrainedModel):
    """
    FastSpeech 2 module.

    This is a module of FastSpeech 2 described in 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech'
    https://arxiv.org/abs/2006.04558. Instead of quantized pitch and energy, we use token-averaged value introduced in
    FastPitch: Parallel Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers instead of regular
    Transformers.
    """
    def __init__(self, config: FastSpeech2ConformerConfig) -> None:
        ...

    @replace_return_docstrings(output_type=FastSpeech2ConformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = ..., spectrogram_labels: Optional[torch.FloatTensor] = ..., duration_labels: Optional[torch.LongTensor] = ..., pitch_labels: Optional[torch.FloatTensor] = ..., energy_labels: Optional[torch.FloatTensor] = ..., speaker_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., speaker_embedding: Optional[torch.FloatTensor] = ..., return_dict: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ...) -> Union[Tuple, FastSpeech2ConformerModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`: 0 for tokens that are **masked**, 1 for tokens that are **not masked**.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*, defaults to `None`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*, defaults to `None`):
                Batch of padded durations.
            pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged pitch.
            energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged energy.
            speaker_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Speaker ids used to condition features of speech output by the model.
            lang_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Language ids used to condition features of speech output by the model.
            speaker_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`, *optional*, defaults to `None`):
                Embedding containing conditioning signals for the features of the speech.
            return_dict (`bool`, *optional*, defaults to `None`):
                Whether or not to return a [`FastSpeech2ConformerModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*, defaults to `None`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*, defaults to `None`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     FastSpeech2ConformerTokenizer,
        ...     FastSpeech2ConformerModel,
        ...     FastSpeech2ConformerHifiGan,
        ... )

        >>> tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        >>> inputs = tokenizer("some text to convert to speech", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]

        >>> model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
        >>> output_dict = model(input_ids, return_dict=True)
        >>> spectrogram = output_dict["spectrogram"]

        >>> vocoder = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
        >>> waveform = vocoder(spectrogram)
        >>> print(waveform.shape)
        torch.Size([1, 49664])
        ```
        """
        ...



class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=..., dilation=..., leaky_relu_slope=...) -> None:
        ...

    def get_padding(self, kernel_size, dilation=...):
        ...

    def apply_weight_norm(self): # -> None:
        ...

    def remove_weight_norm(self): # -> None:
        ...

    def forward(self, hidden_states):
        ...



@add_start_docstrings("""HiFi-GAN vocoder.""", HIFIGAN_START_DOCSTRING)
class FastSpeech2ConformerHifiGan(PreTrainedModel):
    config_class = FastSpeech2ConformerHifiGanConfig
    main_input_name = ...
    def __init__(self, config: FastSpeech2ConformerHifiGanConfig) -> None:
        ...

    def apply_weight_norm(self): # -> None:
        ...

    def remove_weight_norm(self): # -> None:
        ...

    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        ...



@add_start_docstrings("The FastSpeech2ConformerModel with a FastSpeech2ConformerHifiGan vocoder head that performs text-to-speech (waveform).", FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING)
class FastSpeech2ConformerWithHifiGan(PreTrainedModel):
    config_class = FastSpeech2ConformerWithHifiGanConfig
    def __init__(self, config: FastSpeech2ConformerWithHifiGanConfig) -> None:
        ...

    @replace_return_docstrings(output_type=FastSpeech2ConformerWithHifiGanOutput, config_class=FastSpeech2ConformerWithHifiGanConfig)
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = ..., spectrogram_labels: Optional[torch.FloatTensor] = ..., duration_labels: Optional[torch.LongTensor] = ..., pitch_labels: Optional[torch.FloatTensor] = ..., energy_labels: Optional[torch.FloatTensor] = ..., speaker_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., speaker_embedding: Optional[torch.FloatTensor] = ..., return_dict: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ...) -> Union[Tuple, FastSpeech2ConformerModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`: 0 for tokens that are **masked**, 1 for tokens that are **not masked**.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*, defaults to `None`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*, defaults to `None`):
                Batch of padded durations.
            pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged pitch.
            energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged energy.
            speaker_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Speaker ids used to condition features of speech output by the model.
            lang_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Language ids used to condition features of speech output by the model.
            speaker_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`, *optional*, defaults to `None`):
                Embedding containing conditioning signals for the features of the speech.
            return_dict (`bool`, *optional*, defaults to `None`):
                Whether or not to return a [`FastSpeech2ConformerModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*, defaults to `None`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*, defaults to `None`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     FastSpeech2ConformerTokenizer,
        ...     FastSpeech2ConformerWithHifiGan,
        ... )

        >>> tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        >>> inputs = tokenizer("some text to convert to speech", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]

        >>> model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")
        >>> output_dict = model(input_ids, return_dict=True)
        >>> waveform = output_dict["waveform"]
        >>> print(waveform.shape)
        torch.Size([1, 49664])
        ```
        """
        ...

