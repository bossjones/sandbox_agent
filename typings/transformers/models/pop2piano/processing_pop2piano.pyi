"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import List, Optional, Union
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy
from ...utils import TensorType

""" Processor class for Pop2Piano."""
class Pop2PianoProcessor(ProcessorMixin):
    r"""
    Constructs an Pop2Piano processor which wraps a Pop2Piano Feature Extractor and Pop2Piano Tokenizer into a single
    processor.

    [`Pop2PianoProcessor`] offers all the functionalities of [`Pop2PianoFeatureExtractor`] and [`Pop2PianoTokenizer`].
    See the docstring of [`~Pop2PianoProcessor.__call__`] and [`~Pop2PianoProcessor.decode`] for more information.

    Args:
        feature_extractor (`Pop2PianoFeatureExtractor`):
            An instance of [`Pop2PianoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Pop2PianoTokenizer`):
            An instance of ['Pop2PianoTokenizer`]. The tokenizer is a required input.
    """
    attributes = ...
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None:
        ...

    def __call__(self, audio: Union[np.ndarray, List[float], List[np.ndarray]] = ..., sampling_rate: Union[int, List[int]] = ..., steps_per_beat: int = ..., resample: Optional[bool] = ..., notes: Union[List, TensorType] = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., verbose: bool = ..., **kwargs) -> Union[BatchFeature, BatchEncoding]:
        """
        This method uses [`Pop2PianoFeatureExtractor.__call__`] method to prepare log-mel-spectrograms for the model,
        and [`Pop2PianoTokenizer.__call__`] to prepare token_ids from notes.

        Please refer to the docstring of the above two methods for more information.
        """
        ...

    def batch_decode(self, token_ids, feature_extractor_output: BatchFeature, return_midi: bool = ...) -> BatchEncoding:
        """
        This method uses [`Pop2PianoTokenizer.batch_decode`] method to convert model generated token_ids to midi_notes.

        Please refer to the docstring of the above two methods for more information.
        """
        ...

    @property
    def model_input_names(self): # -> list[Any]:
        ...

    def save_pretrained(self, save_directory, **kwargs): # -> list[Any] | list[str]:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs): # -> Self:
        ...

