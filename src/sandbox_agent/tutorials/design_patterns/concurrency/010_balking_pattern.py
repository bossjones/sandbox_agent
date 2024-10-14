"""
Balking Pattern

Overview of the Balking Pattern
The Balking pattern is a design pattern that offers a way to handle a request by an object that is not in a state to deal with it. The Balking pattern can be useful in concurrent programming when a method is called that would normally change the state of an object, but the change is not possible or appropriate at the current time.

In other words, if an object receives a request that it cannot carry out, it will 'balk' at the request, which means it will refuse and return immediately.

Implementing the Balking Pattern in Python
Python's threading module can be used to implement the Balking pattern. Here's an example that uses a thread-safe flag to control whether a task should be run:
"""

from __future__ import annotations

import threading


class BalkingExample:
    def __init__(self):
        self._flag = threading.Event()

    def run_task(self):
        if self._flag.is_set():
            print("Task is already running. Balking...")
            return
        self._flag.set()
        print("Running task...")
        # Simulate a task
        self._flag.clear()


example = BalkingExample()

# Start multiple threads that try to run the task
for _ in range(5):
    threading.Thread(target=example.run_task).start()

"""
In this example, BalkingExample is a class that represents a task that can only be executed once at a time. The _flag attribute is a thread-safe flag that indicates whether the task is currently being executed.

The run_task method checks if the task is already running and if so, it 'balks' and returns immediately. If not, it runs the task.

The Balking pattern provides a way to prevent unnecessary work in situations where an operation cannot or should not be performed concurrently. However, it can introduce complexity into a system, so it should be used judiciously.
"""
