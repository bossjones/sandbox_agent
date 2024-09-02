"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for XLM."""
logger = ...
VOCAB_FILES_NAMES = ...
def get_pairs(word): # -> set[Any]:
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    ...

def lowercase_and_remove_accent(text): # -> list[LiteralString]:
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    ...

def replace_unicode_punct(text): # -> str:
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    ...

def remove_non_printing_char(text): # -> LiteralString:
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    ...

def romanian_preprocessing(text):
    """Sennrich's WMT16 scripts for Romanian preprocessing, used by model `FacebookAI/xlm-mlm-enro-1024`"""
    ...

class XLMTokenizer(PreTrainedTokenizer):
    """
    Construct an XLM tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization for most supported languages.
    - Language specific tokenization for Chinese (Jieba), Japanese (KyTea) and Thai (PyThaiNLP).
    - Optionally lowercases and normalizes all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The `lang2id` attribute maps the languages supported by the model with their IDs if provided (automatically set
      for pretrained vocabularies).
    - The `id2lang` attributes does reverse mapping if provided (automatically set for pretrained vocabularies).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Vocabulary file.
        merges_file (`str`):
            Merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"</s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<special1>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            List of additional special tokens.
        lang2id (`Dict[str, int]`, *optional*):
            Dictionary mapping languages string identifiers to their IDs.
        id2lang (`Dict[int, str]`, *optional*):
            Dictionary mapping language IDs to their string identifiers.
        do_lowercase_and_remove_accent (`bool`, *optional*, defaults to `True`):
            Whether to lowercase and remove accents when tokenizing.
    """
    vocab_files_names = ...
    def __init__(self, vocab_file, merges_file, unk_token=..., bos_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., additional_special_tokens=..., lang2id=..., id2lang=..., do_lowercase_and_remove_accent=..., **kwargs) -> None:
        ...

    @property
    def do_lower_case(self): # -> bool:
        ...

    def moses_punct_norm(self, text, lang):
        ...

    def moses_tokenize(self, text, lang):
        ...

    def moses_pipeline(self, text, lang): # -> LiteralString:
        ...

    def ja_tokenize(self, text): # -> list[Any]:
        ...

    @property
    def vocab_size(self): # -> int:
        ...

    def get_vocab(self): # -> dict[Any, Any]:
        ...

    def bpe(self, token): # -> LiteralString | Literal['\n</w>']:
        ...

    def convert_tokens_to_string(self, tokens): # -> str:
        """Converts a sequence of tokens (string) in a single string."""
        ...

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

        """
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

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLM sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...

    def __getstate__(self): # -> dict[str, Any]:
        ...

    def __setstate__(self, d): # -> None:
        ...

