"""
This type stub file was generated by pyright.
"""

from typing import Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizer

"""Tokenization classes for GPTNeoXJapanese."""
logger = ...
VOCAB_FILES_NAMES = ...
def load_vocab_and_emoji(vocab_file, emoji_file): # -> tuple[OrderedDict[Any, Any], OrderedDict[Any, Any], OrderedDict[Any, Any], Any]:
    """Loads a vocabulary file and emoji file into a dictionary."""
    ...

class GPTNeoXJapaneseTokenizer(PreTrainedTokenizer):
    """
    This tokenizer inherits from [`PreTrainedTokenizer`] and is based on Japanese special Sub-Word-Encoding that is
    used in this repository (https://github.com/tanreinama/Japanese-BPEEncoder_V2). Check the repository for details.
    Japanese has a relatively large vocabulary and there is no separation between words. Furthermore, the language is a
    combination of hiragana, katakana, and kanji, and variants such as "1" and "①" are often used. In order to cope
    with these, this tokenizer has the following features
    - Subword-by-subword segmentation, which is intermediate between byte strings and morphological analysis.
    - BPEs are created for each Kanji, Hiragana, and Katakana character, and there are no BPEs that cross character
        types, such as Kanji + Hiragana or Hiragana + Katakana.
    - All-byte encoding that does not require <unk>.
    - Independent of UTF codes such as 2-byte and 3-byte characters
    - Conversion of heterographs to the same token_id
    - Emoji and Emoticon are grouped into 12 types as special tags.

    Example:

    ```python
    >>> from transformers import GPTNeoXJapaneseTokenizer

    >>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17749
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [30014, 26883, 26638, 27228, 25, 26650, 31732, 31679, 27809, 26638, 17749, 31592, 17749, 31593, 321, 1281]

    >>> # Both 慶応 and 慶應 are decoded to 慶応
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
    """
    vocab_files_names = ...
    model_input_names = ...
    def __init__(self, vocab_file, emoji_file, unk_token=..., pad_token=..., bos_token=..., eos_token=..., do_clean_text=..., **kwargs) -> None:
        ...

    @property
    def vocab_size(self): # -> int:
        ...

    def get_vocab(self): # -> dict[Any, Any]:
        ...

    def convert_tokens_to_string(self, tokens): # -> str:
        """Converts a sequence of tokens (string) in a single string."""
        ...

    @property
    def default_chat_template(self): # -> LiteralString:
        """
        A simple chat template that just adds BOS/EOS tokens around messages while discarding role information.
        """
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...



class SubWordJapaneseTokenizer:
    """
    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT Lisence according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, vocab, ids_to_tokens, emoji) -> None:
        ...

    def __len__(self): # -> int:
        ...

    def clean_text(self, content): # -> str:
        ...

    def tokenize(self, text, clean=...): # -> list[Any]:
        ...

    def convert_id_to_token(self, index, breakline=...): # -> LiteralString:
        ...

