"""
This type stub file was generated by pyright.
"""

import jax.numpy as jnp
from ..utils import add_start_docstrings

logger = ...
LOGITS_PROCESSOR_INPUTS_DOCSTRING = ...
class FlaxLogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for processing logits."""
        ...



class FlaxLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for warping logits."""
        ...



class FlaxLogitsProcessorList(list):
    """
    This class can be used to create a list of [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to subsequently process
    a `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to the inputs.
    """
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs) -> jnp.ndarray:
        ...



class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """
    def __init__(self, temperature: float) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    """
    [`FlaxLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    def __init__(self, top_p: float, filter_value: float = ..., min_tokens_to_keep: int = ...) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    def __init__(self, top_k: int, filter_value: float = ..., min_tokens_to_keep: int = ...) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """
    def __init__(self, bos_token_id: int) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """
    def __init__(self, max_length: int, eos_token_id: int) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """
    def __init__(self, min_length: int, eos_token_id: int) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxSuppressTokensAtBeginLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] supressing a list of tokens as soon as the `generate` function starts generating using
    `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are not sampled at the
    begining of the generation.

    Args:
        begin_suppress_tokens (`List[int]`):
            Tokens to not sample.
        begin_index (`int`):
            Index where the tokens are suppressed.
    """
    def __init__(self, begin_suppress_tokens, begin_index) -> None:
        ...

    def __call__(self, input_ids, scores, cur_len: int):
        ...



class FlaxSuppressTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] suppressing a list of tokens at each decoding step. The processor will set their log probs
    to be `-inf` so they are not sampled.

    Args:
        suppress_tokens (`list`):
            Tokens to not sample.
    """
    def __init__(self, suppress_tokens: list) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxForceTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that takes a list of pairs of integers which indicates a mapping from generation indices to
    token indices that will be forced before sampling. The processor will set their log probs to 0 and all other tokens
    to `-inf` so that they are sampled at their corresponding index.

    Args:
        force_token_map (`list`):
            Map giving token ids and indices where they will be forced to be sampled.
    """
    def __init__(self, force_token_map) -> None:
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...



class FlaxWhisperTimeStampLogitsProcessor(FlaxLogitsProcessor):
    r"""
    Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log
    probs to `inf` so that they are sampled at their corresponding index.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
    """
    def __init__(self, generate_config, model_config, decoder_input_length) -> None:
        ...

    def __call__(self, input_ids, scores, cur_len):
        ...



class FlaxNoRepeatNGramLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """
    def __init__(self, ngram_size: int) -> None:
        ...

    def get_previous_ngrams(self, input_ids: jnp.ndarray, vocab_size: int, cur_len: int):
        """
        get a matrix of size (batch_size,) + (vocab_size,)*n (for n-grams) that
        represent the n-grams that occured previously.
        The BCOO representation allow to store only the few non-zero entries, instead of the full (huge) matrix
        """
        ...

    def get_banned_tokens_mask(self, latest_tokens: jnp.ndarray, previous_ngrams) -> jnp.ndarray:
        """
        Determines which tokens must be banned given latest tokens and the previously seen
        ngrams.
        """
        ...

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        ...

