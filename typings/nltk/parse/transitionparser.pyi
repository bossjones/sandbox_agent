"""
This type stub file was generated by pyright.
"""

from nltk.parse import ParserI

class Configuration:
    """
    Class for holding configuration which is the partial analysis of the input sentence.
    The transition based parser aims at finding set of operators that transfer the initial
    configuration to the terminal configuration.

    The configuration includes:
        - Stack: for storing partially proceeded words
        - Buffer: for storing remaining input words
        - Set of arcs: for storing partially built dependency tree

    This class also provides a method to represent a configuration as list of features.
    """
    def __init__(self, dep_graph) -> None:
        """
        :param dep_graph: the representation of an input in the form of dependency graph.
        :type dep_graph: DependencyGraph where the dependencies are not specified.
        """
        ...

    def __str__(self) -> str:
        ...

    def extract_features(self): # -> list:
        """
        Extract the set of features for the current configuration. Implement standard features as describe in
        Table 3.2 (page 31) in Dependency Parsing book by Sandra Kubler, Ryan McDonal, Joakim Nivre.
        Please note that these features are very basic.
        :return: list(str)
        """
        ...



class Transition:
    """
    This class defines a set of transition which is applied to a configuration to get another configuration
    Note that for different parsing algorithm, the transition is different.
    """
    LEFT_ARC = ...
    RIGHT_ARC = ...
    SHIFT = ...
    REDUCE = ...
    def __init__(self, alg_option) -> None:
        """
        :param alg_option: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type alg_option: str
        """
        ...

    def left_arc(self, conf, relation): # -> Literal[-1] | None:
        """
        Note that the algorithm for left-arc is quite similar except for precondition for both arc-standard and arc-eager

        :param configuration: is the current configuration
        :return: A new configuration or -1 if the pre-condition is not satisfied
        """
        ...

    def right_arc(self, conf, relation): # -> Literal[-1] | None:
        """
        Note that the algorithm for right-arc is DIFFERENT for arc-standard and arc-eager

        :param configuration: is the current configuration
        :return: A new configuration or -1 if the pre-condition is not satisfied
        """
        ...

    def reduce(self, conf): # -> Literal[-1] | None:
        """
        Note that the algorithm for reduce is only available for arc-eager

        :param configuration: is the current configuration
        :return: A new configuration or -1 if the pre-condition is not satisfied
        """
        ...

    def shift(self, conf): # -> Literal[-1] | None:
        """
        Note that the algorithm for shift is the SAME for arc-standard and arc-eager

        :param configuration: is the current configuration
        :return: A new configuration or -1 if the pre-condition is not satisfied
        """
        ...



class TransitionParser(ParserI):
    """
    Class for transition based parser. Implement 2 algorithms which are "arc-standard" and "arc-eager"
    """
    ARC_STANDARD = ...
    ARC_EAGER = ...
    def __init__(self, algorithm) -> None:
        """
        :param algorithm: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type algorithm: str
        """
        ...

    def train(self, depgraphs, modelfile, verbose=...): # -> None:
        """
        :param depgraphs : list of DependencyGraph as the training data
        :type depgraphs : DependencyGraph
        :param modelfile : file name to save the trained model
        :type modelfile : str
        """
        ...

    def parse(self, depgraphs, modelFile): # -> list:
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        ...



def demo(): # -> None:
    """
    >>> from nltk.parse import DependencyGraph, DependencyEvaluator
    >>> from nltk.parse.transitionparser import TransitionParser, Configuration, Transition
    >>> gold_sent = DependencyGraph(\"""
    ... Economic  JJ     2      ATT
    ... news  NN     3       SBJ
    ... has       VBD       0       ROOT
    ... little      JJ      5       ATT
    ... effect   NN     3       OBJ
    ... on     IN      5       ATT
    ... financial       JJ       8       ATT
    ... markets    NNS      6       PC
    ... .    .      3       PU
    ... \""")

    >>> conf = Configuration(gold_sent)

    ###################### Check the Initial Feature ########################

    >>> print(', '.join(conf.extract_features()))
    STK_0_POS_TOP, BUF_0_FORM_Economic, BUF_0_LEMMA_Economic, BUF_0_POS_JJ, BUF_1_FORM_news, BUF_1_POS_NN, BUF_2_POS_VBD, BUF_3_POS_JJ

    ###################### Check The Transition #######################
    Check the Initialized Configuration
    >>> print(conf)
    Stack : [0]  Buffer : [1, 2, 3, 4, 5, 6, 7, 8, 9]   Arcs : []

    A. Do some transition checks for ARC-STANDARD

    >>> operation = Transition('arc-standard')
    >>> operation.shift(conf)
    >>> operation.left_arc(conf, "ATT")
    >>> operation.shift(conf)
    >>> operation.left_arc(conf,"SBJ")
    >>> operation.shift(conf)
    >>> operation.shift(conf)
    >>> operation.left_arc(conf, "ATT")
    >>> operation.shift(conf)
    >>> operation.shift(conf)
    >>> operation.shift(conf)
    >>> operation.left_arc(conf, "ATT")

    Middle Configuration and Features Check
    >>> print(conf)
    Stack : [0, 3, 5, 6]  Buffer : [8, 9]   Arcs : [(2, 'ATT', 1), (3, 'SBJ', 2), (5, 'ATT', 4), (8, 'ATT', 7)]

    >>> print(', '.join(conf.extract_features()))
    STK_0_FORM_on, STK_0_LEMMA_on, STK_0_POS_IN, STK_1_POS_NN, BUF_0_FORM_markets, BUF_0_LEMMA_markets, BUF_0_POS_NNS, BUF_1_FORM_., BUF_1_POS_., BUF_0_LDEP_ATT

    >>> operation.right_arc(conf, "PC")
    >>> operation.right_arc(conf, "ATT")
    >>> operation.right_arc(conf, "OBJ")
    >>> operation.shift(conf)
    >>> operation.right_arc(conf, "PU")
    >>> operation.right_arc(conf, "ROOT")
    >>> operation.shift(conf)

    Terminated Configuration Check
    >>> print(conf)
    Stack : [0]  Buffer : []   Arcs : [(2, 'ATT', 1), (3, 'SBJ', 2), (5, 'ATT', 4), (8, 'ATT', 7), (6, 'PC', 8), (5, 'ATT', 6), (3, 'OBJ', 5), (3, 'PU', 9), (0, 'ROOT', 3)]


    B. Do some transition checks for ARC-EAGER

    >>> conf = Configuration(gold_sent)
    >>> operation = Transition('arc-eager')
    >>> operation.shift(conf)
    >>> operation.left_arc(conf,'ATT')
    >>> operation.shift(conf)
    >>> operation.left_arc(conf,'SBJ')
    >>> operation.right_arc(conf,'ROOT')
    >>> operation.shift(conf)
    >>> operation.left_arc(conf,'ATT')
    >>> operation.right_arc(conf,'OBJ')
    >>> operation.right_arc(conf,'ATT')
    >>> operation.shift(conf)
    >>> operation.left_arc(conf,'ATT')
    >>> operation.right_arc(conf,'PC')
    >>> operation.reduce(conf)
    >>> operation.reduce(conf)
    >>> operation.reduce(conf)
    >>> operation.right_arc(conf,'PU')
    >>> print(conf)
    Stack : [0, 3, 9]  Buffer : []   Arcs : [(2, 'ATT', 1), (3, 'SBJ', 2), (0, 'ROOT', 3), (5, 'ATT', 4), (3, 'OBJ', 5), (5, 'ATT', 6), (8, 'ATT', 7), (6, 'PC', 8), (3, 'PU', 9)]

    ###################### Check The Training Function #######################

    A. Check the ARC-STANDARD training
    >>> import tempfile
    >>> import os
    >>> input_file = tempfile.NamedTemporaryFile(prefix='transition_parse.train', dir=tempfile.gettempdir(), delete=False)

    >>> parser_std = TransitionParser('arc-standard')
    >>> print(', '.join(parser_std._create_training_examples_arc_std([gold_sent], input_file)))
     Number of training examples : 1
     Number of valid (projective) examples : 1
    SHIFT, LEFTARC:ATT, SHIFT, LEFTARC:SBJ, SHIFT, SHIFT, LEFTARC:ATT, SHIFT, SHIFT, SHIFT, LEFTARC:ATT, RIGHTARC:PC, RIGHTARC:ATT, RIGHTARC:OBJ, SHIFT, RIGHTARC:PU, RIGHTARC:ROOT, SHIFT

    >>> parser_std.train([gold_sent],'temp.arcstd.model', verbose=False)
     Number of training examples : 1
     Number of valid (projective) examples : 1
    >>> input_file.close()
    >>> remove(input_file.name)

    B. Check the ARC-EAGER training

    >>> input_file = tempfile.NamedTemporaryFile(prefix='transition_parse.train', dir=tempfile.gettempdir(),delete=False)
    >>> parser_eager = TransitionParser('arc-eager')
    >>> print(', '.join(parser_eager._create_training_examples_arc_eager([gold_sent], input_file)))
     Number of training examples : 1
     Number of valid (projective) examples : 1
    SHIFT, LEFTARC:ATT, SHIFT, LEFTARC:SBJ, RIGHTARC:ROOT, SHIFT, LEFTARC:ATT, RIGHTARC:OBJ, RIGHTARC:ATT, SHIFT, LEFTARC:ATT, RIGHTARC:PC, REDUCE, REDUCE, REDUCE, RIGHTARC:PU

    >>> parser_eager.train([gold_sent],'temp.arceager.model', verbose=False)
     Number of training examples : 1
     Number of valid (projective) examples : 1

    >>> input_file.close()
    >>> remove(input_file.name)

    ###################### Check The Parsing Function ########################

    A. Check the ARC-STANDARD parser

    >>> result = parser_std.parse([gold_sent], 'temp.arcstd.model')
    >>> de = DependencyEvaluator(result, [gold_sent])
    >>> de.eval() >= (0, 0)
    True

    B. Check the ARC-EAGER parser
    >>> result = parser_eager.parse([gold_sent], 'temp.arceager.model')
    >>> de = DependencyEvaluator(result, [gold_sent])
    >>> de.eval() >= (0, 0)
    True

    Remove test temporary files
    >>> remove('temp.arceager.model')
    >>> remove('temp.arcstd.model')

    Note that result is very poor because of only one training example.
    """
    ...
