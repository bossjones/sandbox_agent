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
    def template_method(self):
        self.prepare_data()
        self.analyze_data()
        self.present_results()

    def prepare_data(self):
        print("Preparation stage might be common for all subclasses")

    @abstractmethod
    def analyze_data(self):
        pass

    @abstractmethod
    def present_results(self):
        pass


class ConcreteDataMiner1(DataMiner):
    def analyze_data(self):
        print("Data analysis by ConcreteDataMiner1")

    def present_results(self):
        print("Results presentation by ConcreteDataMiner1")


class ConcreteDataMiner2(DataMiner):
    def analyze_data(self):
        print("Data analysis by ConcreteDataMiner2")

    def present_results(self):
        print("Results presentation by ConcreteDataMiner2")


# Client code
miner1 = ConcreteDataMiner1()
miner1.template_method()

miner2 = ConcreteDataMiner2()
miner2.template_method()

"""
In this example, DataMiner is our abstract class implementing the template method,
and ConcreteDataMiner1 and ConcreteDataMiner2 are our concrete classes that implement the abstract methods defined in DataMiner.

The Template Method pattern provides a way to define a skeleton of an algorithm in a base class and let the subclasses redefine certain steps.
It promotes code reuse and isolates the complex or volatile parts of an algorithm to the subclasses. However, clients might have limited control over the algorithm
since they can't override the template method itself.
"""
