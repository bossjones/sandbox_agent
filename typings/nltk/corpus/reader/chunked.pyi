"""
This type stub file was generated by pyright.
"""

from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *

"""
A reader for corpora that contain chunked (and optionally tagged)
documents.
"""
class ChunkedCorpusReader(CorpusReader):
    """
    Reader for chunked (and optionally tagged) corpora.  Paragraphs
    are split using a block reader.  They are then tokenized into
    sentences using a sentence tokenizer.  Finally, these sentences
    are parsed into chunk trees using a string-to-chunktree conversion
    function.  Each of these steps can be performed using a default
    function or a custom function.  By default, paragraphs are split
    on blank lines; sentences are listed one per line; and sentences
    are parsed into chunk trees using ``nltk.chunk.tagstr2tree``.
    """
    def __init__(self, root, fileids, extension=..., str2chunktree=..., sent_tokenizer=..., para_block_reader=..., encoding=..., tagset=...) -> None:
        """
        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        """
        ...

    def words(self, fileids=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        ...

    def sents(self, fileids=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))
        """
        ...

    def paras(self, fileids=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        ...

    def tagged_words(self, fileids=..., tagset=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of tagged
            words and punctuation symbols, encoded as tuples
            ``(word,tag)``.
        :rtype: list(tuple(str,str))
        """
        ...

    def tagged_sents(self, fileids=..., tagset=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a list of ``(word,tag)`` tuples.

        :rtype: list(list(tuple(str,str)))
        """
        ...

    def tagged_paras(self, fileids=..., tagset=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of ``(word,tag)`` tuples.
        :rtype: list(list(list(tuple(str,str))))
        """
        ...

    def chunked_words(self, fileids=..., tagset=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of tagged
            words and chunks.  Words are encoded as ``(word, tag)``
            tuples (if the corpus has tags) or word strings (if the
            corpus has no tags).  Chunks are encoded as depth-one
            trees over ``(word,tag)`` tuples or word strings.
        :rtype: list(tuple(str,str) and Tree)
        """
        ...

    def chunked_sents(self, fileids=..., tagset=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a shallow Tree.  The leaves
            of these trees are encoded as ``(word, tag)`` tuples (if
            the corpus has tags) or word strings (if the corpus has no
            tags).
        :rtype: list(Tree)
        """
        ...

    def chunked_paras(self, fileids=..., tagset=...): # -> str | ConcatenatedCorpusView | LazyConcatenation | list | tuple[()] | Element:
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as a shallow Tree.  The leaves of these
            trees are encoded as ``(word, tag)`` tuples (if the corpus
            has tags) or word strings (if the corpus has no tags).
        :rtype: list(list(Tree))
        """
        ...



class ChunkedCorpusView(StreamBackedCorpusView):
    def __init__(self, fileid, encoding, tagged, group_by_sent, group_by_para, chunked, str2chunktree, sent_tokenizer, para_block_reader, source_tagset=..., target_tagset=...) -> None:
        ...

    def read_block(self, stream): # -> list:
        ...
