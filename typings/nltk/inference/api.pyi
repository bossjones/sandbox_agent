"""
This type stub file was generated by pyright.
"""

import threading
from abc import ABCMeta, abstractmethod

"""
Interfaces and base classes for theorem provers and model builders.

``Prover`` is a standard interface for a theorem prover which tries to prove a goal from a
list of assumptions.

``ModelBuilder`` is a standard interface for a model builder. Given just a set of assumptions.
the model builder tries to build a model for the assumptions. Given a set of assumptions and a
goal *G*, the model builder tries to find a counter-model, in the sense of a model that will satisfy
the assumptions plus the negation of *G*.
"""
class Prover(metaclass=ABCMeta):
    """
    Interface for trying to prove a goal from assumptions.  Both the goal and
    the assumptions are constrained to be formulas of ``logic.Expression``.
    """
    def prove(self, goal=..., assumptions=..., verbose=...):
        """
        :return: Whether the proof was successful or not.
        :rtype: bool
        """
        ...



class ModelBuilder(metaclass=ABCMeta):
    """
    Interface for trying to build a model of set of formulas.
    Open formulas are assumed to be universally quantified.
    Both the goal and the assumptions are constrained to be formulas
    of ``logic.Expression``.
    """
    def build_model(self, goal=..., assumptions=..., verbose=...):
        """
        Perform the actual model building.
        :return: Whether a model was generated
        :rtype: bool
        """
        ...



class TheoremToolCommand(metaclass=ABCMeta):
    """
    This class holds a goal and a list of assumptions to be used in proving
    or model building.
    """
    @abstractmethod
    def add_assumptions(self, new_assumptions): # -> None:
        """
        Add new assumptions to the assumption list.

        :param new_assumptions: new assumptions
        :type new_assumptions: list(sem.Expression)
        """
        ...

    @abstractmethod
    def retract_assumptions(self, retracted, debug=...): # -> None:
        """
        Retract assumptions from the assumption list.

        :param debug: If True, give warning when ``retracted`` is not present on
            assumptions list.
        :type debug: bool
        :param retracted: assumptions to be retracted
        :type retracted: list(sem.Expression)
        """
        ...

    @abstractmethod
    def assumptions(self): # -> None:
        """
        List the current assumptions.

        :return: list of ``Expression``
        """
        ...

    @abstractmethod
    def goal(self): # -> None:
        """
        Return the goal

        :return: ``Expression``
        """
        ...

    @abstractmethod
    def print_assumptions(self): # -> None:
        """
        Print the list of the current assumptions.
        """
        ...



class ProverCommand(TheoremToolCommand):
    """
    This class holds a ``Prover``, a goal, and a list of assumptions.  When
    prove() is called, the ``Prover`` is executed with the goal and assumptions.
    """
    @abstractmethod
    def prove(self, verbose=...): # -> None:
        """
        Perform the actual proof.
        """
        ...

    @abstractmethod
    def proof(self, simplify=...): # -> None:
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    @abstractmethod
    def get_prover(self): # -> None:
        """
        Return the prover object
        :return: ``Prover``
        """
        ...



class ModelBuilderCommand(TheoremToolCommand):
    """
    This class holds a ``ModelBuilder``, a goal, and a list of assumptions.
    When build_model() is called, the ``ModelBuilder`` is executed with the goal
    and assumptions.
    """
    @abstractmethod
    def build_model(self, verbose=...): # -> None:
        """
        Perform the actual model building.
        :return: A model if one is generated; None otherwise.
        :rtype: sem.Valuation
        """
        ...

    @abstractmethod
    def model(self, format=...): # -> None:
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    @abstractmethod
    def get_model_builder(self): # -> None:
        """
        Return the model builder object
        :return: ``ModelBuilder``
        """
        ...



class BaseTheoremToolCommand(TheoremToolCommand):
    """
    This class holds a goal and a list of assumptions to be used in proving
    or model building.
    """
    def __init__(self, goal=..., assumptions=...) -> None:
        """
        :param goal: Input expression to prove
        :type goal: sem.Expression
        :param assumptions: Input expressions to use as assumptions in
            the proof.
        :type assumptions: list(sem.Expression)
        """
        ...

    def add_assumptions(self, new_assumptions): # -> None:
        """
        Add new assumptions to the assumption list.

        :param new_assumptions: new assumptions
        :type new_assumptions: list(sem.Expression)
        """
        ...

    def retract_assumptions(self, retracted, debug=...): # -> None:
        """
        Retract assumptions from the assumption list.

        :param debug: If True, give warning when ``retracted`` is not present on
            assumptions list.
        :type debug: bool
        :param retracted: assumptions to be retracted
        :type retracted: list(sem.Expression)
        """
        ...

    def assumptions(self): # -> list:
        """
        List the current assumptions.

        :return: list of ``Expression``
        """
        ...

    def goal(self): # -> None:
        """
        Return the goal

        :return: ``Expression``
        """
        ...

    def print_assumptions(self): # -> None:
        """
        Print the list of the current assumptions.
        """
        ...



class BaseProverCommand(BaseTheoremToolCommand, ProverCommand):
    """
    This class holds a ``Prover``, a goal, and a list of assumptions.  When
    prove() is called, the ``Prover`` is executed with the goal and assumptions.
    """
    def __init__(self, prover, goal=..., assumptions=...) -> None:
        """
        :param prover: The theorem tool to execute with the assumptions
        :type prover: Prover
        :see: ``BaseTheoremToolCommand``
        """
        ...

    def prove(self, verbose=...):
        """
        Perform the actual proof.  Store the result to prevent unnecessary
        re-proving.
        """
        ...

    def proof(self, simplify=...): # -> None:
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    def decorate_proof(self, proof_string, simplify=...):
        """
        Modify and return the proof string
        :param proof_string: str the proof to decorate
        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    def get_prover(self): # -> Any:
        ...



class BaseModelBuilderCommand(BaseTheoremToolCommand, ModelBuilderCommand):
    """
    This class holds a ``ModelBuilder``, a goal, and a list of assumptions.  When
    build_model() is called, the ``ModelBuilder`` is executed with the goal and
    assumptions.
    """
    def __init__(self, modelbuilder, goal=..., assumptions=...) -> None:
        """
        :param modelbuilder: The theorem tool to execute with the assumptions
        :type modelbuilder: ModelBuilder
        :see: ``BaseTheoremToolCommand``
        """
        ...

    def build_model(self, verbose=...):
        """
        Attempt to build a model.  Store the result to prevent unnecessary
        re-building.
        """
        ...

    def model(self, format=...): # -> None:
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    def get_model_builder(self): # -> Any:
        ...



class TheoremToolCommandDecorator(TheoremToolCommand):
    """
    A base decorator for the ``ProverCommandDecorator`` and
    ``ModelBuilderCommandDecorator`` classes from which decorators can extend.
    """
    def __init__(self, command) -> None:
        """
        :param command: ``TheoremToolCommand`` to decorate
        """
        ...

    def assumptions(self):
        ...

    def goal(self):
        ...

    def add_assumptions(self, new_assumptions): # -> None:
        ...

    def retract_assumptions(self, retracted, debug=...): # -> None:
        ...

    def print_assumptions(self): # -> None:
        ...



class ProverCommandDecorator(TheoremToolCommandDecorator, ProverCommand):
    """
    A base decorator for the ``ProverCommand`` class from which other
    prover command decorators can extend.
    """
    def __init__(self, proverCommand) -> None:
        """
        :param proverCommand: ``ProverCommand`` to decorate
        """
        ...

    def prove(self, verbose=...):
        ...

    def proof(self, simplify=...):
        """
        Return the proof string
        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    def decorate_proof(self, proof_string, simplify=...):
        """
        Modify and return the proof string
        :param proof_string: str the proof to decorate
        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    def get_prover(self):
        ...



class ModelBuilderCommandDecorator(TheoremToolCommandDecorator, ModelBuilderCommand):
    """
    A base decorator for the ``ModelBuilderCommand`` class from which other
    prover command decorators can extend.
    """
    def __init__(self, modelBuilderCommand) -> None:
        """
        :param modelBuilderCommand: ``ModelBuilderCommand`` to decorate
        """
        ...

    def build_model(self, verbose=...):
        """
        Attempt to build a model.  Store the result to prevent unnecessary
        re-building.
        """
        ...

    def model(self, format=...):
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """
        ...

    def get_model_builder(self):
        ...



class ParallelProverBuilder(Prover, ModelBuilder):
    """
    This class stores both a prover and a model builder and when either
    prove() or build_model() is called, then both theorem tools are run in
    parallel.  Whichever finishes first, the prover or the model builder, is the
    result that will be used.
    """
    def __init__(self, prover, modelbuilder) -> None:
        ...



class ParallelProverBuilderCommand(BaseProverCommand, BaseModelBuilderCommand):
    """
    This command stores both a prover and a model builder and when either
    prove() or build_model() is called, then both theorem tools are run in
    parallel.  Whichever finishes first, the prover or the model builder, is the
    result that will be used.

    Because the theorem prover result is the opposite of the model builder
    result, we will treat self._result as meaning "proof found/no model found".
    """
    def __init__(self, prover, modelbuilder, goal=..., assumptions=...) -> None:
        ...

    def prove(self, verbose=...): # -> bool:
        ...

    def build_model(self, verbose=...): # -> bool:
        ...



class TheoremToolThread(threading.Thread):
    def __init__(self, command, verbose, name=...) -> None:
        ...

    def run(self): # -> None:
        ...

    @property
    def result(self): # -> None:
        ...
