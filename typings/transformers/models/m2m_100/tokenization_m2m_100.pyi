"""
This type stub file was generated by pyright.
"""

import sentencepiece
from typing import Any, Dict, List, Optional, Tuple, Union
from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer

"""Tokenization classes for M2M100."""
logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...
FAIRSEQ_LANGUAGE_CODES = ...
class M2M100Tokenizer(PreTrainedTokenizer):
    """
    Construct an M2M100 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        spm_file (`str`):
            Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        language_codes (`str`, *optional*, defaults to `"m2m100"`):
            What language codes to use. Should be one of `"m2m100"` or `"wmt21"`.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Examples:

    ```python
    >>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    >>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    >>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="ro")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    >>> outputs = model(**model_inputs)  # should work
    ```"""
    vocab_files_names = ...
    model_input_names = ...
    prefix_tokens: List[int] = ...
    suffix_tokens: List[int] = ...
    def __init__(self, vocab_file, spm_file, src_lang=..., tgt_lang=..., bos_token=..., eos_token=..., sep_token=..., pad_token=..., unk_token=..., language_codes=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., num_madeup_words=..., **kwargs) -> None:
        ...

    @property
    def vocab_size(self) -> int:
        ...

    def get_vocab(self) -> Dict:
        ...

    @property
    def src_lang(self) -> str:
        ...

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        ...

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        ...

    def __getstate__(self) -> Dict:
        ...

    def __setstate__(self, d: Dict) -> None:
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...

    def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str = ..., tgt_texts: Optional[List[str]] = ..., tgt_lang: str = ..., **kwargs) -> BatchEncoding:
        ...

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        ...

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        ...

    def get_lang_token(self, lang: str) -> str:
        ...

    def get_lang_id(self, lang: str) -> int:
        ...



def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    ...

def load_json(path: str) -> Union[Dict, List]:
    ...

def save_json(data, path: str) -> None:
    ...
