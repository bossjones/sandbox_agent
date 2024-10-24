"""
This type stub file was generated by pyright.
"""

from nltk.tag import ClassifierBasedTagger
from nltk.chunk.api import ChunkParserI

"""
Named entity chunker
"""
class NEChunkParserTagger(ClassifierBasedTagger):
    """
    The IOB tagger used by the chunk parser.
    """
    def __init__(self, train=..., classifier=...) -> None:
        ...



class NEChunkParser(ChunkParserI):
    """
    Expected input: list of pos-tagged words
    """
    def __init__(self, train) -> None:
        ...

    def parse(self, tokens): # -> Tree:
        """
        Each token should be a pos-tagged word
        """
        ...



def shape(word): # -> Literal['number', 'punct', 'upcase', 'downcase', 'mixedcase', 'other']:
    ...

def simplify_pos(s): # -> Literal['V']:
    ...

def postag_tree(tree): # -> Tree:
    ...

def load_ace_data(roots, fmt=..., skip_bnews=...): # -> Generator[Tree, Any, None]:
    ...

def load_ace_file(textfile, fmt): # -> Generator[Tree, Any, None]:
    ...

def cmp_chunks(correct, guessed): # -> None:
    ...

class Maxent_NE_Chunker(NEChunkParser):
    """
    Expected input: list of pos-tagged words
    """
    def __init__(self, fmt=...) -> None:
        ...

    def load_params(self): # -> None:
        ...

    def save_params(self): # -> None:
        ...



def build_model(fmt=...): # -> Maxent_NE_Chunker:
    ...

if __name__ == "__main__":
    ...