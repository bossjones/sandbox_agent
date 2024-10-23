# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Strategy Pattern

Overview of the Strategy Pattern

The Strategy pattern is a behavioral design pattern that lets you define a family of algorithms, put each of them
into a separate class, and make their objects interchangeable. The main goal of the Strategy pattern is to enable a
client to choose between different algorithms or procedures to complete a specific task. It allows you to switch the
algorithm or strategy based upon changes in requirements.

Implementing the Strategy Pattern in Python

Let's consider an example where we have a Traveler class, and the traveler can choose different modes of
transportation. The choice of transport is what varies, so we can use the Strategy pattern to select the
transportation method at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TravelStrategy(ABC):
    @abstractmethod
    def travel_to(self, destination: str) -> str:
        """
        Travel to the given destination.

        Args:
            destination (str): The destination to travel to.

        Returns:
            str: The message indicating the mode of travel to the destination.
        """
        pass


class AirTravelStrategy(TravelStrategy):
    def travel_to(self, destination: str) -> str:
        """
        Travel to the given destination by air.

        Args:
            destination (str): The destination to travel to.

        Returns:
            str: The message indicating traveling to the destination by air.
        """
        return f"Traveling to {destination} by air."


class RailTravelStrategy(TravelStrategy):
    def travel_to(self, destination: str) -> str:
        """
        Travel to the given destination by rail.

        Args:
            destination (str): The destination to travel to.

        Returns:
            str: The message indicating traveling to the destination by rail.
        """
        return f"Traveling to {destination} by rail."


class RoadTravelStrategy(TravelStrategy):
    def travel_to(self, destination: str) -> str:
        """
        Travel to the given destination by road.

        Args:
            destination (str): The destination to travel to.

        Returns:
            str: The message indicating traveling to the destination by road.
        """
        return f"Traveling to {destination} by road."


class Traveler:
    def __init__(self, strategy: TravelStrategy):
        """
        Initialize the Traveler with the given TravelStrategy.

        Args:
            strategy (TravelStrategy): The travel strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: TravelStrategy) -> None:
        """
        Set the travel strategy for the Traveler.

        Args:
            strategy (TravelStrategy): The new travel strategy to use.
        """
        self._strategy = strategy

    def travel(self, destination: str) -> None:
        """
        Travel to the given destination using the current travel strategy.

        Args:
            destination (str): The destination to travel to.
        """
        print(self._strategy.travel_to(destination))


# Client code
traveler = Traveler(AirTravelStrategy())
traveler.travel("Paris")  # Outputs: Traveling to Paris by air.

traveler.set_strategy(RailTravelStrategy())
traveler.travel("Berlin")  # Outputs: Traveling to Berlin by rail.

traveler.set_strategy(RoadTravelStrategy())
traveler.travel("Rome")  # Outputs: Traveling to Rome by road.


# In this example, TravelStrategy is an abstract base class that defines an interface for all strategies.
# AirTravelStrategy, RailTravelStrategy, and RoadTravelStrategy are concrete strategies that implement the
# TravelStrategy interface. The Traveler class uses a TravelStrategy to define how it will travel to a destination.

# The Strategy pattern allows you to switch out the "guts" of an object at runtime, thus promoting flexibility in
# your design. It encapsulates the algorithm implementations from the clients that use them and makes them easy to
# switch, understand, and extend. However, it can complicate the code by introducing additional classes and can be
# overkill for a design if the strategy classes are very similar and differ only in the way they deliver results.
