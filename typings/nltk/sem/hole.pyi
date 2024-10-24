"""
This type stub file was generated by pyright.
"""

"""
An implementation of the Hole Semantics model, following Blackburn and Bos,
Representation and Inference for Natural Language (CSLI, 2005).

The semantic representations are built by the grammar hole.fcfg.
This module contains driver code to read in sentences and parse them
according to a hole semantics grammar.

After parsing, the semantic representation is in the form of an underspecified
representation that is not easy to read.  We use a "plugging" algorithm to
convert that representation into first-order logic formulas.
"""
class Constants:
    ALL = ...
    EXISTS = ...
    NOT = ...
    AND = ...
    OR = ...
    IMP = ...
    IFF = ...
    PRED = ...
    LEQ = ...
    HOLE = ...
    LABEL = ...
    MAP = ...


class HoleSemantics:
    """
    This class holds the broken-down components of a hole semantics, i.e. it
    extracts the holes, labels, logic formula fragments and constraints out of
    a big conjunction of such as produced by the hole semantics grammar.  It
    then provides some operations on the semantics dealing with holes, labels
    and finding legal ways to plug holes with labels.
    """
    def __init__(self, usr) -> None:
        """
        Constructor.  `usr' is a ``sem.Expression`` representing an
        Underspecified Representation Structure (USR).  A USR has the following
        special predicates:
        ALL(l,v,n),
        EXISTS(l,v,n),
        AND(l,n,n),
        OR(l,n,n),
        IMP(l,n,n),
        IFF(l,n,n),
        PRED(l,v,n,v[,v]*) where the brackets and star indicate zero or more repetitions,
        LEQ(n,n),
        HOLE(n),
        LABEL(n)
        where l is the label of the node described by the predicate, n is either
        a label or a hole, and v is a variable.
        """
        ...

    def is_node(self, x): # -> bool:
        """
        Return true if x is a node (label or hole) in this semantic
        representation.
        """
        ...

    def pluggings(self): # -> list:
        """
        Calculate and return all the legal pluggings (mappings of labels to
        holes) of this semantics given the constraints.
        """
        ...

    def formula_tree(self, plugging):
        """
        Return the first-order logic formula tree for this underspecified
        representation using the plugging given.
        """
        ...



class Constraint:
    """
    This class represents a constraint of the form (L =< N),
    where L is a label and N is a node (a label or a hole).
    """
    def __init__(self, lhs, rhs) -> None:
        ...

    def __eq__(self, other) -> bool:
        ...

    def __ne__(self, other) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __repr__(self): # -> str:
        ...



def hole_readings(sentence, grammar_filename=..., verbose=...): # -> list:
    ...

if __name__ == "__main__":
    ...