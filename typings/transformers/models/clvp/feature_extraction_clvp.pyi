"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import List, Optional, Union
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType

"""
Feature extractor class for CLVP
"""
logger = ...
class ClvpFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLVP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts log-mel-spectrogram features from raw speech using a custom numpy implementation of the `Short
    Time Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 22050):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        default_audio_length (`int`, *optional*, defaults to 6):
            The default length of raw audio in seconds. If `max_length` is not set during `__call__` then it will
            automatically be set to default_audio_length * `self.sampling_rate`.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        mel_norms (`list` of length `feature_size`, *optional*):
            If `mel_norms` is provided then it will be used to normalize the log-mel spectrograms along each
            mel-filter.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. If left to the default, it will return the attention mask.

            [What are attention masks?](../glossary#attention-mask)
    """
    model_input_names = ...
    def __init__(self, feature_size=..., sampling_rate=..., default_audio_length=..., hop_length=..., chunk_length=..., n_fft=..., padding_value=..., mel_norms=..., return_attention_mask=..., **kwargs) -> None:
        ...

    def __call__(self, raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], sampling_rate: Optional[int] = ..., truncation: bool = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_attention_mask: Optional[bool] = ..., padding: Optional[str] = ..., max_length: Optional[int] = ..., **kwargs) -> BatchFeature:
        """
        `ClvpFeatureExtractor` is used to extract various voice specific properties such as the pitch and tone of the
        voice, speaking speed, and even speaking defects like a lisp or stuttering from a sample voice or `raw_speech`.

        First the voice is padded or truncated in a way such that it becomes a waveform of `self.default_audio_length`
        seconds long and then the log-mel spectrogram is extracted from it.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return the attention mask. If left to the default, it will return the attention mask.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
            max_length (`int`, *optional*):
                The maximum input length of the inputs.
        """
        ...

