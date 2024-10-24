"""
This type stub file was generated by pyright.
"""

from typing import Tuple
from nltk.stem.api import StemmerI

class Cistem(StemmerI):
    """
    CISTEM Stemmer for German

    This is the official Python implementation of the CISTEM stemmer.
    It is based on the paper
    Leonie Weissweiler, Alexander Fraser (2017). Developing a Stemmer for German
    Based on a Comparative Analysis of Publicly Available Stemmers.
    In Proceedings of the German Society for Computational Linguistics and Language
    Technology (GSCL)
    which can be read here:
    https://www.cis.lmu.de/~weissweiler/cistem/

    In the paper, we conducted an analysis of publicly available stemmers,
    developed two gold standards for German stemming and evaluated the stemmers
    based on the two gold standards. We then proposed the stemmer implemented here
    and show that it achieves slightly better f-measure than the other stemmers and
    is thrice as fast as the Snowball stemmer for German while being about as fast
    as most other stemmers.

    case_insensitive is a a boolean specifying if case-insensitive stemming
    should be used. Case insensitivity improves performance only if words in the
    text may be incorrectly upper case. For all-lowercase and correctly cased
    text, best performance is achieved by setting case_insensitive for false.

    :param case_insensitive: if True, the stemming is case insensitive. False by default.
    :type case_insensitive: bool
    """
    strip_ge = ...
    repl_xx = ...
    strip_emr = ...
    strip_nd = ...
    strip_t = ...
    strip_esn = ...
    repl_xx_back = ...
    def __init__(self, case_insensitive: bool = ...) -> None:
        ...

    @staticmethod
    def replace_to(word: str) -> str:
        ...

    @staticmethod
    def replace_back(word: str) -> str:
        ...

    def stem(self, word: str) -> str:
        """Stems the input word.

        :param word: The word that is to be stemmed.
        :type word: str
        :return: The stemmed word.
        :rtype: str

        >>> from nltk.stem.cistem import Cistem
        >>> stemmer = Cistem()
        >>> s1 = "Speicherbehältern"
        >>> stemmer.stem(s1)
        'speicherbehalt'
        >>> s2 = "Grenzpostens"
        >>> stemmer.stem(s2)
        'grenzpost'
        >>> s3 = "Ausgefeiltere"
        >>> stemmer.stem(s3)
        'ausgefeilt'
        >>> stemmer = Cistem(True)
        >>> stemmer.stem(s1)
        'speicherbehal'
        >>> stemmer.stem(s2)
        'grenzpo'
        >>> stemmer.stem(s3)
        'ausgefeil'
        """
        ...

    def segment(self, word: str) -> Tuple[str, str]:
        """
        This method works very similarly to stem (:func:'cistem.stem'). The difference is that in
        addition to returning the stem, it also returns the rest that was removed at
        the end. To be able to return the stem unchanged so the stem and the rest
        can be concatenated to form the original word, all subsitutions that altered
        the stem in any other way than by removing letters at the end were left out.

        :param word: The word that is to be stemmed.
        :type word: str
        :return: A tuple of the stemmed word and the removed suffix.
        :rtype: Tuple[str, str]

        >>> from nltk.stem.cistem import Cistem
        >>> stemmer = Cistem()
        >>> s1 = "Speicherbehältern"
        >>> stemmer.segment(s1)
        ('speicherbehält', 'ern')
        >>> s2 = "Grenzpostens"
        >>> stemmer.segment(s2)
        ('grenzpost', 'ens')
        >>> s3 = "Ausgefeiltere"
        >>> stemmer.segment(s3)
        ('ausgefeilt', 'ere')
        >>> stemmer = Cistem(True)
        >>> stemmer.segment(s1)
        ('speicherbehäl', 'tern')
        >>> stemmer.segment(s2)
        ('grenzpo', 'stens')
        >>> stemmer.segment(s3)
        ('ausgefeil', 'tere')
        """
        ...