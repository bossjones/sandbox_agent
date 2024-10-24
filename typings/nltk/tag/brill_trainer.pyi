"""
This type stub file was generated by pyright.
"""

class BrillTaggerTrainer:
    """
    A trainer for tbl taggers.
    """
    def __init__(self, initial_tagger, templates, trace=..., deterministic=..., ruleformat=...) -> None:
        """
        Construct a Brill tagger from a baseline tagger and a
        set of templates

        :param initial_tagger: the baseline tagger
        :type initial_tagger: Tagger
        :param templates: templates to be used in training
        :type templates: list of Templates
        :param trace: verbosity level
        :type trace: int
        :param deterministic: if True, adjudicate ties deterministically
        :type deterministic: bool
        :param ruleformat: format of reported Rules
        :type ruleformat: str
        :return: An untrained BrillTagger
        :rtype: BrillTagger
        """
        ...

    def train(self, train_sents, max_rules=..., min_score=..., min_acc=...): # -> BrillTagger:
        r"""
        Trains the Brill tagger on the corpus *train_sents*,
        producing at most *max_rules* transformations, each of which
        reduces the net number of errors in the corpus by at least
        *min_score*, and each of which has accuracy not lower than
        *min_acc*.

        >>> # Relevant imports
        >>> from nltk.tbl.template import Template
        >>> from nltk.tag.brill import Pos, Word
        >>> from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer

        >>> # Load some data
        >>> from nltk.corpus import treebank
        >>> training_data = treebank.tagged_sents()[:100]
        >>> baseline_data = treebank.tagged_sents()[100:200]
        >>> gold_data = treebank.tagged_sents()[200:300]
        >>> testing_data = [untag(s) for s in gold_data]

        >>> backoff = RegexpTagger([
        ... (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
        ... (r'(The|the|A|a|An|an)$', 'AT'),   # articles
        ... (r'.*able$', 'JJ'),                # adjectives
        ... (r'.*ness$', 'NN'),                # nouns formed from adjectives
        ... (r'.*ly$', 'RB'),                  # adverbs
        ... (r'.*s$', 'NNS'),                  # plural nouns
        ... (r'.*ing$', 'VBG'),                # gerunds
        ... (r'.*ed$', 'VBD'),                 # past tense verbs
        ... (r'.*', 'NN')                      # nouns (default)
        ... ])

        >>> baseline = backoff #see NOTE1
        >>> baseline.accuracy(gold_data) #doctest: +ELLIPSIS
        0.243...

        >>> # Set up templates
        >>> Template._cleartemplates() #clear any templates created in earlier tests
        >>> templates = [Template(Pos([-1])), Template(Pos([-1]), Word([0]))]

        >>> # Construct a BrillTaggerTrainer
        >>> tt = BrillTaggerTrainer(baseline, templates, trace=3)

        >>> tagger1 = tt.train(training_data, max_rules=10)
        TBL train (fast) (seqs: 100; tokens: 2417; tpls: 2; min score: 2; min acc: None)
        Finding initial useful rules...
            Found 847 useful rules.
        <BLANKLINE>
                   B      |
           S   F   r   O  |        Score = Fixed - Broken
           c   i   o   t  |  R     Fixed = num tags changed incorrect -> correct
           o   x   k   h  |  u     Broken = num tags changed correct -> incorrect
           r   e   e   e  |  l     Other = num tags changed incorrect -> incorrect
           e   d   n   r  |  e
        ------------------+-------------------------------------------------------
         132 132   0   0  | AT->DT if Pos:NN@[-1]
          85  85   0   0  | NN->, if Pos:NN@[-1] & Word:,@[0]
          69  69   0   0  | NN->. if Pos:NN@[-1] & Word:.@[0]
          51  51   0   0  | NN->IN if Pos:NN@[-1] & Word:of@[0]
          47  63  16 162  | NN->IN if Pos:NNS@[-1]
          33  33   0   0  | NN->TO if Pos:NN@[-1] & Word:to@[0]
          26  26   0   0  | IN->. if Pos:NNS@[-1] & Word:.@[0]
          24  24   0   0  | IN->, if Pos:NNS@[-1] & Word:,@[0]
          22  27   5  24  | NN->-NONE- if Pos:VBD@[-1]
          17  17   0   0  | NN->CC if Pos:NN@[-1] & Word:and@[0]

        >>> tagger1.rules()[1:3]
        (Rule('001', 'NN', ',', [(Pos([-1]),'NN'), (Word([0]),',')]), Rule('001', 'NN', '.', [(Pos([-1]),'NN'), (Word([0]),'.')]))

        >>> train_stats = tagger1.train_stats()
        >>> [train_stats[stat] for stat in ['initialerrors', 'finalerrors', 'rulescores']]
        [1776, 1270, [132, 85, 69, 51, 47, 33, 26, 24, 22, 17]]

        >>> tagger1.print_template_statistics(printunused=False)
        TEMPLATE STATISTICS (TRAIN)  2 templates, 10 rules)
        TRAIN (   2417 tokens) initial  1776 0.2652 final:  1270 0.4746
        #ID | Score (train) |  #Rules     | Template
        --------------------------------------------
        001 |   305   0.603 |   7   0.700 | Template(Pos([-1]),Word([0]))
        000 |   201   0.397 |   3   0.300 | Template(Pos([-1]))
        <BLANKLINE>
        <BLANKLINE>

        >>> round(tagger1.accuracy(gold_data),5)
        0.43834

        >>> tagged, test_stats = tagger1.batch_tag_incremental(testing_data, gold_data)

        >>> tagged[33][12:] == [('foreign', 'IN'), ('debt', 'NN'), ('of', 'IN'), ('$', 'NN'), ('64', 'CD'),
        ... ('billion', 'NN'), ('*U*', 'NN'), ('--', 'NN'), ('the', 'DT'), ('third-highest', 'NN'), ('in', 'NN'),
        ... ('the', 'DT'), ('developing', 'VBG'), ('world', 'NN'), ('.', '.')]
        True

        >>> [test_stats[stat] for stat in ['initialerrors', 'finalerrors', 'rulescores']]
        [1859, 1380, [100, 85, 67, 58, 27, 36, 27, 16, 31, 32]]

        >>> # A high-accuracy tagger
        >>> tagger2 = tt.train(training_data, max_rules=10, min_acc=0.99)
        TBL train (fast) (seqs: 100; tokens: 2417; tpls: 2; min score: 2; min acc: 0.99)
        Finding initial useful rules...
            Found 847 useful rules.
        <BLANKLINE>
                   B      |
           S   F   r   O  |        Score = Fixed - Broken
           c   i   o   t  |  R     Fixed = num tags changed incorrect -> correct
           o   x   k   h  |  u     Broken = num tags changed correct -> incorrect
           r   e   e   e  |  l     Other = num tags changed incorrect -> incorrect
           e   d   n   r  |  e
        ------------------+-------------------------------------------------------
         132 132   0   0  | AT->DT if Pos:NN@[-1]
          85  85   0   0  | NN->, if Pos:NN@[-1] & Word:,@[0]
          69  69   0   0  | NN->. if Pos:NN@[-1] & Word:.@[0]
          51  51   0   0  | NN->IN if Pos:NN@[-1] & Word:of@[0]
          36  36   0   0  | NN->TO if Pos:NN@[-1] & Word:to@[0]
          26  26   0   0  | NN->. if Pos:NNS@[-1] & Word:.@[0]
          24  24   0   0  | NN->, if Pos:NNS@[-1] & Word:,@[0]
          19  19   0   6  | NN->VB if Pos:TO@[-1]
          18  18   0   0  | CD->-NONE- if Pos:NN@[-1] & Word:0@[0]
          18  18   0   0  | NN->CC if Pos:NN@[-1] & Word:and@[0]

        >>> round(tagger2.accuracy(gold_data), 8)
        0.43996744

        >>> tagger2.rules()[2:4]
        (Rule('001', 'NN', '.', [(Pos([-1]),'NN'), (Word([0]),'.')]), Rule('001', 'NN', 'IN', [(Pos([-1]),'NN'), (Word([0]),'of')]))

        # NOTE1: (!!FIXME) A far better baseline uses nltk.tag.UnigramTagger,
        # with a RegexpTagger only as backoff. For instance,
        # >>> baseline = UnigramTagger(baseline_data, backoff=backoff)
        # However, as of Nov 2013, nltk.tag.UnigramTagger does not yield consistent results
        # between python versions. The simplistic backoff above is a workaround to make doctests
        # get consistent input.

        :param train_sents: training data
        :type train_sents: list(list(tuple))
        :param max_rules: output at most max_rules rules
        :type max_rules: int
        :param min_score: stop training when no rules better than min_score can be found
        :type min_score: int
        :param min_acc: discard any rule with lower accuracy than min_acc
        :type min_acc: float or None
        :return: the learned tagger
        :rtype: BrillTagger
        """
        ...