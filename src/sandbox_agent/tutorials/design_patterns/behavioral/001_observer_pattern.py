# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Observer Pattern

Overview of the Observer Pattern The Observer pattern is a design pattern that defines a one-to-many dependency between objects so that when one object changes state, all of its dependents are notified and updated automatically. The object that watches on the state of another object is called Observer, and the object that is being watched is called Subject. Implementing the Observer Pattern in Python Let's take an example where we have a WeatherStation (Subject) that tracks weather conditions. We can have multiple Display elements (Observers) that need to update whenever the WeatherStation updates. Here's how you can implement the Observer pattern in Python:
"""
# Here is the extracted text from the image:

# ---
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Set


class Observer(ABC):
    """
    Abstract base class for observers.
    """

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Update the observer with new data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass


class DisplayElement(ABC):
    """
    Abstract base class for display elements.
    """

    @abstractmethod
    def display(self) -> None:
        """
        Display the current state.
        """
        pass


class WeatherStation:
    """
    Weather station class that acts as the subject.
    """

    def __init__(self) -> None:
        """
        Initialize the WeatherStation.
        """
        self._observers: set[Observer] = set()
        self._temperature: float | None = None
        self._humidity: float | None = None
        self._pressure: float | None = None

    def register_observer(self, observer: Observer) -> None:
        """
        Register an observer.

        Args:
            observer: The observer to register.
        """
        observer._subject = self  # type: ignore
        self._observers.add(observer)

    def remove_observer(self, observer: Observer) -> None:
        """
        Remove an observer.

        Args:
            observer: The observer to remove.
        """
        observer._subject = None  # type: ignore
        self._observers.discard(observer)

    def notify_observers(self) -> None:
        """
        Notify all observers about an event.
        """
        for observer in self._observers:
            observer.update(self._temperature, self._humidity, self._pressure)

    def measurements_changed(self) -> None:
        """
        Indicate that measurements have changed and notify observers.
        """
        self.notify_observers()

    def set_measurements(self, temperature: float, humidity: float, pressure: float) -> None:
        """
        Set new measurements and notify observers.

        Args:
            temperature: The temperature measurement.
            humidity: The humidity measurement.
            pressure: The pressure measurement.
        """
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self.measurements_changed()


class CurrentConditionsDisplay(Observer, DisplayElement):
    """
    Current conditions display class that acts as an observer.
    """

    def __init__(self, weather_station: WeatherStation) -> None:
        """
        Initialize the CurrentConditionsDisplay.

        Args:
            weather_station: The weather station to observe.
        """
        self._subject = weather_station
        self._temperature: float | None = None
        self._humidity: float | None = None
        self._subject.register_observer(self)

    def update(self, temperature: float | None, humidity: float | None, pressure: float | None) -> None:
        """
        Update the display with new measurements.

        Args:
            temperature: The temperature measurement.
            humidity: The humidity measurement.
            pressure: The pressure measurement (unused).
        """
        self._temperature = temperature
        self._humidity = humidity
        self.display()

    def display(self) -> None:
        """
        Display the current conditions.
        """
        print(f"Current conditions: {self._temperature}F degrees and {self._humidity}% humidity")


# Client code
weather_station = WeatherStation()
current_display = CurrentConditionsDisplay(weather_station)
weather_station.set_measurements(80, 65, 30.4)


# In this example, **WeatherStation** is the **Subject** and **CurrentConditionsDisplay** is an **Observer**. **WeatherStation** notifies **CurrentConditionsDisplay** whenever it updates its measurements.

# The **Observer pattern** is beneficial when a change to one object requires changing others, and you don't know how many objects need to be changed. It provides a loosely coupled design between objects that interact, which is flexible and can easily accommodate future changes. However, it can lead to unexpected updates if not managed carefully, and the order of delivery isn't guaranteed.

# ---

# Let me know if you need further clarification!
