"""
This type stub file was generated by pyright.
"""

from ...processing_utils import ProcessorMixin

"""Speech processor class for SpeechT5."""
class SpeechT5Processor(ProcessorMixin):
    r"""
    Constructs a SpeechT5 processor which wraps a feature extractor and a tokenizer into a single processor.

    [`SpeechT5Processor`] offers all the functionalities of [`SpeechT5FeatureExtractor`] and [`SpeechT5Tokenizer`]. See
    the docstring of [`~SpeechT5Processor.__call__`] and [`~SpeechT5Processor.decode`] for more information.

    Args:
        feature_extractor (`SpeechT5FeatureExtractor`):
            An instance of [`SpeechT5FeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`SpeechT5Tokenizer`):
            An instance of [`SpeechT5Tokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = ...
    tokenizer_class = ...
    def __init__(self, feature_extractor, tokenizer) -> None:
        ...

    def __call__(self, *args, **kwargs): # -> None:
        """
        Processes audio and text input, as well as audio and text targets.

        You can process audio by using the argument `audio`, or process audio targets by using the argument
        `audio_target`. This forwards the arguments to SpeechT5FeatureExtractor's
        [`~SpeechT5FeatureExtractor.__call__`].

        You can process text by using the argument `text`, or process text labels by using the argument `text_target`.
        This forwards the arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.__call__`].

        Valid input combinations are:

        - `text` only
        - `audio` only
        - `text_target` only
        - `audio_target` only
        - `text` and `audio_target`
        - `audio` and `audio_target`
        - `text` and `text_target`
        - `audio` and `text_target`

        Please refer to the docstring of the above two methods for more information.
        """
        ...

    def pad(self, *args, **kwargs): # -> None:
        """
        Collates the audio and text inputs, as well as their targets, into a padded batch.

        Audio inputs are padded by SpeechT5FeatureExtractor's [`~SpeechT5FeatureExtractor.pad`]. Text inputs are padded
        by SpeechT5Tokenizer's [`~SpeechT5Tokenizer.pad`].

        Valid input combinations are:

        - `input_ids` only
        - `input_values` only
        - `labels` only, either log-mel spectrograms or text tokens
        - `input_ids` and log-mel spectrogram `labels`
        - `input_values` and text `labels`

        Please refer to the docstring of the above two methods for more information.
        """
        ...

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        ...

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        ...

