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


class Observer:
    def update(self, *args, **kwargs):
        pass


class DisplayElement:
    def display(self):
        pass


class WeatherStation:
    def __init__(self):
        self._observers = set()
        self._temperature = None
        self._humidity = None
        self._pressure = None

    def register_observer(self, observer):
        observer._subject = self
        self._observers.add(observer)

    def remove_observer(self, observer):
        observer._subject = None
        self._observers.discard(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.update(self._temperature, self._humidity, self._pressure)

    def measurements_changed(self):
        self.notify_observers()

    def set_measurements(self, temperature, humidity, pressure):
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self.measurements_changed()


class CurrentConditionsDisplay(Observer, DisplayElement):
    def __init__(self, weather_station):
        self._subject = weather_station
        self._subject.register_observer(self)

    def update(self, temperature, humidity, pressure):
        self._temperature = temperature
        self._humidity = humidity
        self.display()

    def display(self):
        print(f"Current conditions: {self._temperature}F degrees and {self._humidity}% humidity")


# Client code
weather_station = WeatherStation()
current_display = CurrentConditionsDisplay(weather_station)
weather_station.set_measurements(80, 65, 30.4)


# In this example, **WeatherStation** is the **Subject** and **CurrentConditionsDisplay** is an **Observer**. **WeatherStation** notifies **CurrentConditionsDisplay** whenever it updates its measurements.

# The **Observer pattern** is beneficial when a change to one object requires changing others, and you don't know how many objects need to be changed. It provides a loosely coupled design between objects that interact, which is flexible and can easily accommodate future changes. However, it can lead to unexpected updates if not managed carefully, and the order of delivery isn't guaranteed.

# ---

# Let me know if you need further clarification!
