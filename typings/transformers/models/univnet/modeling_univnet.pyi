"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_univnet import UnivNetConfig

""" PyTorch UnivNetModel model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
@dataclass
class UnivNetModelOutput(ModelOutput):
    """
    Output class for the [`UnivNetModel`], which includes the generated audio waveforms and the original unpadded
    lengths of those waveforms (so that the padding can be removed by [`UnivNetModel.batch_decode`]).

    Args:
        waveforms (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Batched 1D (mono-channel) output audio waveforms.
        waveform_lengths (`torch.FloatTensor` of shape `(batch_size,)`):
            The batched length in samples of each unpadded waveform in `waveforms`.
    """
    waveforms: torch.FloatTensor = ...
    waveform_lengths: torch.FloatTensor = ...


class UnivNetKernelPredictorResidualBlock(nn.Module):
    """
    Implementation of the residual block for the kernel predictor network inside each location variable convolution
    block (LVCBlock).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
    """
    def __init__(self, config: UnivNetConfig) -> None:
        ...

    def forward(self, hidden_states: torch.FloatTensor): # -> Tensor:
        ...

    def apply_weight_norm(self): # -> None:
        ...

    def remove_weight_norm(self): # -> None:
        ...



class UnivNetKernelPredictor(nn.Module):
    """
    Implementation of the kernel predictor network which supplies the kernel and bias for the location variable
    convolutional layers (LVCs) in each UnivNet LVCBlock.

    Based on the KernelPredictor implementation in
    [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L7).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the location variable convolutional layer kernels (convolutional weight tensor).
        conv_layers (`int`, *optional*, defaults to 4):
            The number of location variable convolutional layers to output kernels and biases for.
    """
    def __init__(self, config: UnivNetConfig, conv_kernel_size: int = ..., conv_layers: int = ...) -> None:
        ...

    def forward(self, spectrogram: torch.FloatTensor): # -> tuple[Any, Any]:
        """
        Maps a conditioning log-mel spectrogram to a tensor of convolutional kernels and biases, for use in location
        variable convolutional layers. Note that the input spectrogram should have shape (batch_size, input_channels,
        seq_length).

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, input_channels, seq_length)`):
                Tensor containing the log-mel spectrograms.

        Returns:
            Tuple[`torch.FloatTensor, `torch.FloatTensor`]: tuple of tensors where the first element is the tensor of
            location variable convolution kernels of shape `(batch_size, self.conv_layers, self.conv_in_channels,
            self.conv_out_channels, self.conv_kernel_size, seq_length)` and the second element is the tensor of
            location variable convolution biases of shape `(batch_size, self.conv_layers. self.conv_out_channels,
            seq_length)`.
        """
        ...

    def apply_weight_norm(self): # -> None:
        ...

    def remove_weight_norm(self): # -> None:
        ...



class UnivNetLvcResidualBlock(nn.Module):
    """
    Implementation of the location variable convolution (LVC) residual block for the UnivNet residual network.

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        kernel_size (`int`):
            The kernel size for the dilated 1D convolutional layer.
        dilation (`int`):
            The dilation for the dilated 1D convolutional layer.
    """
    def __init__(self, config: UnivNetConfig, kernel_size: int, dilation: int) -> None:
        ...

    def forward(self, hidden_states, kernel, bias, hop_size=...):
        ...

    def location_variable_convolution(self, hidden_states: torch.FloatTensor, kernel: torch.FloatTensor, bias: torch.FloatTensor, dilation: int = ..., hop_size: int = ...): # -> Tensor:
        """
        Performs location-variable convolution operation on the input sequence (hidden_states) using the local
        convolution kernel. This was introduced in [LVCNet: Efficient Condition-Dependent Modeling Network for Waveform
        Generation](https://arxiv.org/abs/2102.10815) by Zhen Zheng, Jianzong Wang, Ning Cheng, and Jing Xiao.

        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, in_channels, in_length)`):
                The input sequence of shape (batch, in_channels, in_length).
            kernel (`torch.FloatTensor` of shape `(batch_size, in_channels, out_channels, kernel_size, kernel_length)`):
                The local convolution kernel of shape (batch, in_channels, out_channels, kernel_size, kernel_length).
            bias (`torch.FloatTensor` of shape `(batch_size, out_channels, kernel_length)`):
                The bias for the local convolution of shape (batch, out_channels, kernel_length).
            dilation (`int`, *optional*, defaults to 1):
                The dilation of convolution.
            hop_size (`int`, *optional*, defaults to 256):
                The hop_size of the conditioning sequence.
        Returns:
            `torch.FloatTensor`: the output sequence after performing local convolution with shape (batch_size,
            out_channels, in_length).
        """
        ...

    def apply_weight_norm(self): # -> None:
        ...

    def remove_weight_norm(self): # -> None:
        ...



class UnivNetLvcBlock(nn.Module):
    """
    Implementation of the location variable convolution (LVC) residual block of the UnivNet residual block. Includes a
    `UnivNetKernelPredictor` inside to predict the kernels and biases of the LVC layers.

    Based on LVCBlock in
    [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L98)

    Parameters:
        config (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        layer_id (`int`):
            An integer corresponding to the index of the current LVC resnet block layer. This should be between 0 and
            `len(config.resblock_stride_sizes) - 1)` inclusive.
        lvc_hop_size (`int`, *optional*, defaults to 256):
            The hop size for the location variable convolutional layers.
    """
    def __init__(self, config: UnivNetConfig, layer_id: int, lvc_hop_size: int = ...) -> None:
        ...

    def forward(self, hidden_states: torch.FloatTensor, spectrogram: torch.FloatTensor): # -> FloatTensor:
        ...

    def apply_weight_norm(self): # -> None:
        ...

    def remove_weight_norm(self): # -> None:
        ...



UNIVNET_START_DOCSTRING = ...
UNIVNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("""UnivNet GAN vocoder.""", UNIVNET_START_DOCSTRING)
class UnivNetModel(PreTrainedModel):
    config_class = UnivNetConfig
    main_input_name = ...
    def __init__(self, config: UnivNetConfig) -> None:
        ...

    @add_start_docstrings_to_model_forward(UNIVNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UnivNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_features: torch.FloatTensor, noise_sequence: Optional[torch.FloatTensor] = ..., padding_mask: Optional[torch.FloatTensor] = ..., generator: Optional[torch.Generator] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], UnivNetModelOutput]:
        r"""
        Returns:

        Example:

         ```python
         >>> from transformers import UnivNetFeatureExtractor, UnivNetModel
         >>> from datasets import load_dataset, Audio

         >>> model = UnivNetModel.from_pretrained("dg845/univnet-dev")
         >>> feature_extractor = UnivNetFeatureExtractor.from_pretrained("dg845/univnet-dev")

         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> # Resample the audio to the feature extractor's sampling rate.
         >>> ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
         >>> inputs = feature_extractor(
         ...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
         ... )
         >>> audio = model(**inputs).waveforms
         >>> list(audio.shape)
         [1, 140288]
         ```
        """
        ...

    def apply_weight_norm(self): # -> None:
        ...

    def remove_weight_norm(self): # -> None:
        ...

