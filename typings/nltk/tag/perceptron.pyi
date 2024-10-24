"""
This type stub file was generated by pyright.
"""

from nltk import jsontags
from nltk.tag.api import TaggerI

TRAINED_TAGGER_PATH = ...
TAGGER_JSONS = ...
@jsontags.register_tag
class AveragedPerceptron:
    """An averaged perceptron, as implemented by Matthew Honnibal.

    See more implementation details here:
        https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
    """
    json_tag = ...
    def __init__(self, weights=...) -> None:
        ...

    def predict(self, features, return_conf=...): # -> tuple[Any, Any | None]:
        """Dot-product the features and current weights and return the best label."""
        ...

    def update(self, truth, guess, features): # -> None:
        """Update the feature weights."""
        ...

    def average_weights(self): # -> None:
        """Average weights from all iterations."""
        ...

    def save(self, path): # -> None:
        """Save the model weights as json"""
        ...

    def load(self, path): # -> None:
        """Load the json model weights."""
        ...

    def encode_json_obj(self): # -> Any | dict:
        ...

    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...



@jsontags.register_tag
class PerceptronTagger(TaggerI):
    """
    Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
    https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

    >>> from nltk.tag.perceptron import PerceptronTagger

    Train the model

    >>> tagger = PerceptronTagger(load=False)

    >>> tagger.train([[('today','NN'),('is','VBZ'),('good','JJ'),('day','NN')],
    ... [('yes','NNS'),('it','PRP'),('beautiful','JJ')]])

    >>> tagger.tag(['today','is','a','beautiful','day'])
    [('today', 'NN'), ('is', 'PRP'), ('a', 'PRP'), ('beautiful', 'JJ'), ('day', 'NN')]

    Use the pretrain model (the default constructor)

    >>> pretrain = PerceptronTagger()

    >>> pretrain.tag('The quick brown fox jumps over the lazy dog'.split())
    [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

    >>> pretrain.tag("The red cat".split())
    [('The', 'DT'), ('red', 'JJ'), ('cat', 'NN')]
    """
    json_tag = ...
    START = ...
    END = ...
    def __init__(self, load=..., lang=...) -> None:
        """
        :param load: Load the json model upon instantiation.
        """
        ...

    def tag(self, tokens, return_conf=..., use_tagdict=...): # -> list:
        """
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        """
        ...

    def train(self, sentences, save_loc=..., nr_iter=...): # -> None:
        """Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a json model in this location.
        :param nr_iter: Number of training iterations.
        """
        ...

    def save_to_json(self, loc, lang=...): # -> None:
        ...

    def load_from_json(self, lang=...): # -> None:
        ...

    def encode_json_obj(self): # -> tuple[Any | dict, Any | dict, list[Any]]:
        ...

    @classmethod
    def decode_json_obj(cls, obj): # -> Self:
        ...

    def normalize(self, word): # -> Literal['!HYPHEN', '!YEAR', '!DIGITS']:
        """
        Normalization used in pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        """
        ...



if __name__ == "__main__":
    ...