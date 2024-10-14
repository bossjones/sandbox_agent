"""
Creating a GUI Toolkit using the Factory Method

The Factory Method is a design pattern that provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created. This pattern is particularly useful when a class cannot anticipate the type of objects it needs to create.

In this section, we will use the Factory Method pattern to create a simple GUI (Graphical User Interface) toolkit.

Concept of the GUI Toolkit with Factory Method

The main idea is to have a base WidgetFactory class with a factory method create_widget(). This method is used to create a widget, but the specific type of widget is determined by the concrete factory classes. For instance, we can have a ButtonFactory that creates button widgets and a LabelFactory that creates label widgets.
"""

# Code Example in Python

# We'll use the tkinter module to create the GUI elements:
from __future__ import annotations

import tkinter as tk

from abc import ABC, abstractmethod


class WidgetFactory(ABC):
    @abstractmethod
    def create_widget(self):
        pass


class ButtonFactory(WidgetFactory):
    def create_widget(self):
        return tk.Button(text="I'm a button")


class LabelFactory(WidgetFactory):
    def create_widget(self):
        return tk.Label(text="I'm a label")


def draw_gui(factory):
    widget = factory.create_widget()
    widget.pack()


if __name__ == "__main__":
    root = tk.Tk()
    draw_gui(ButtonFactory())
    draw_gui(LabelFactory())
    root.mainloop()

"""
In this code:
- WidgetFactory is the abstract base class that defines the factory method create_widget().
- ButtonFactory and LabelFactory are subclasses that provide concrete implementations for the factory method. Each creates a different type of widget: a button and a label, respectively.
- draw_gui() is a function that accepts a WidgetFactory, creates a widget using the factory method, and adds the widget to the GUI.

When we run this code, we see a simple GUI with a button and a label. This is a basic example, but it illustrates how the Factory Method pattern allows us to create objects without specifying their exact classes. With this pattern, we can easily extend our GUI toolkit to support more widget types by adding more factory classes.
"""
