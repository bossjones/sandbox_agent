# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
"""
Command pattern decouples the object invoking a job from the one who knows
how to do it. As mentioned in the GoF book, a good example is in menu items.
You have a menu that has lots of items. Each item is responsible for doing a
special thing and you want your menu item just call the execute method when
it is pressed. To achieve this you implement a command object with the execute
method for each menu item and pass to it.

*About the example
We have a menu containing two items. Each item accepts a file name, one hides the file
and the other deletes it. Both items have an undo option.
Each item is a MenuItem class that accepts the corresponding command as input and executes
it's execute method when it is pressed.

*TL;DR
Object oriented implementation of callback functions.

*Examples in Python ecosystem:
Django HttpRequest (without execute method):
https://docs.djangoproject.com/en/2.1/ref/request-response/#httprequest-objects
"""

"""
Command Pattern
Overview of the Command Pattern

The Command pattern is a behavioral design pattern that turns a request into a stand-alone object that contains all information about the request.
The transformation lets you pass requests as method arguments, delay or queue a request's execution, and support undoable operations.
It provides a way to decouple the sender of a request from its receiver by giving multiple objects a chance to handle the request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        """
        Execute the command.
        """
        pass


class Light:
    def turn_on(self) -> None:
        """
        Turn on the light.
        """
        print("The light is on")

    def turn_off(self) -> None:
        """
        Turn off the light.
        """
        print("The light is off")

    def dim(self) -> None:
        """
        Dim the light.
        """
        print("The light is dimmed")


class TurnOnCommand(Command):
    def __init__(self, light: Light):
        """
        Initialize the TurnOnCommand with the given Light.

        Args:
            light (Light): The light to control.
        """
        self._light = light

    def execute(self) -> None:
        """
        Execute the turn on command.
        """
        self._light.turn_on()


class TurnOffCommand(Command):
    def __init__(self, light: Light):
        """
        Initialize the TurnOffCommand with the given Light.

        Args:
            light (Light): The light to control.
        """
        self._light = light

    def execute(self) -> None:
        """
        Execute the turn off command.
        """
        self._light.turn_off()


class DimCommand(Command):
    def __init__(self, light: Light):
        """
        Initialize the DimCommand with the given Light.

        Args:
            light (Light): The light to control.
        """
        self._light = light

    def execute(self) -> None:
        """
        Execute the dim command.
        """
        self._light.dim()


class RemoteControl:
    def submit(self, command: Command) -> None:
        """
        Submit a command to be executed.

        Args:
            command (Command): The command to submit.
        """
        command.execute()


# Client code
light = Light()
remote = RemoteControl()

remote.submit(TurnOnCommand(light))  # Outputs: The light is on
remote.submit(DimCommand(light))  # Outputs: The light is dimmed
remote.submit(TurnOffCommand(light))  # Outputs: The light is off

"""
In this example, Command is an abstract base class with an execute method.
TurnOnCommand, TurnOffCommand, and DimCommand are concrete commands that implement the Command interface.
Light is our receiver, and RemoteControl is our invoker.

The Command pattern can encapsulate a request as an object, thereby letting you parameterize other objects with different requests, operations, or both.
It supports undoable operations, and it can support logging changes so that they can be reapplied in case of a crash.
However, the Command pattern can lead to a high number of command classes.
"""
