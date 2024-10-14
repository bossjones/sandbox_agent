# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Template Method Pattern

Overview of the Template Method Pattern

The Template Method pattern is a behavioral design pattern that defines the program skeleton of an algorithm
in a superclass but lets subclasses override specific steps of the algorithm without changing its structure.
The template method itself should be made final to prevent further modifications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DataMiner(ABC):
    """
    Abstract base class representing a data miner.
    """

    def template_method(self) -> None:
        """
        The template method defines the skeleton of the data mining algorithm.
        """
        self.prepare_data()
        self.analyze_data()
        self.present_results()

    def prepare_data(self) -> None:
        """
        Prepare the data for analysis.
        """
        print("Preparation stage might be common for all subclasses")

    @abstractmethod
    def analyze_data(self) -> None:
        """
        Analyze the data (abstract method to be implemented by subclasses).
        """
        pass

    @abstractmethod
    def present_results(self) -> None:
        """
        Present the results of the data analysis (abstract method to be implemented by subclasses).
        """
        pass


class ConcreteDataMiner1(DataMiner):
    """
    Concrete subclass of DataMiner implementing specific steps of the algorithm.
    """

    def analyze_data(self) -> None:
        """
        Analyze the data specific to ConcreteDataMiner1.
        """
        print("Data analysis by ConcreteDataMiner1")

    def present_results(self) -> None:
        """
        Present the results specific to ConcreteDataMiner1.
        """
        print("Results presentation by ConcreteDataMiner1")


class ConcreteDataMiner2(DataMiner):
    """
    Concrete subclass of DataMiner implementing specific steps of the algorithm.
    """

    def analyze_data(self) -> None:
        """
        Analyze the data specific to ConcreteDataMiner2.
        """
        print("Data analysis by ConcreteDataMiner2")

    def present_results(self) -> None:
        """
        Present the results specific to ConcreteDataMiner2.
        """
        print("Results presentation by ConcreteDataMiner2")


# Client code
miner1 = ConcreteDataMiner1()
miner1.template_method()
# Output:
# Preparation stage might be common for all subclasses
# Data analysis by ConcreteDataMiner1
# Results presentation by ConcreteDataMiner1

miner2 = ConcreteDataMiner2()
miner2.template_method()
# Output:
# Preparation stage might be common for all subclasses
# Data analysis by ConcreteDataMiner2
# Results presentation by ConcreteDataMiner2

"""
In this example, DataMiner is our abstract class implementing the template method,
and ConcreteDataMiner1 and ConcreteDataMiner2 are our concrete classes that implement the abstract methods defined in DataMiner.

The Template Method pattern provides a way to define a skeleton of an algorithm in a base class and let the subclasses redefine certain steps.
It promotes code reuse and isolates the complex or volatile parts of an algorithm to the subclasses. However, clients might have limited control over the algorithm
since they can't override the template method itself.
"""
