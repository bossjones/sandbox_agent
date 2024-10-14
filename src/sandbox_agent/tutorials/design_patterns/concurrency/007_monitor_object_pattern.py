"""
Monitor Object Pattern

Overview of the Monitor Object Pattern
The Monitor Object pattern is a design pattern that synchronizes method execution, ensuring that only one method at a time runs within an object. It also allows an object's methods to cooperatively schedule their execution sequences. This pattern can be viewed as a box which contains an object's procedures and data, and that provides mutual exclusion for threads that are executing procedures of the object.

The Monitor Object pattern is used to protect an object's state from inconsistent and undesirable changes that could occur when the object is accessed concurrently by multiple threads.

Implementing the Monitor Object Pattern in Python
In Python, the threading module provides the Lock class, which can be used to create monitor objects. Here's an example of how to use it to implement a thread-safe stack:
"""

from __future__ import annotations

import threading


class Stack:
    def __init__(self):
        self.stack = []
        self.lock = threading.Lock()

    def push(self, item):
        with self.lock:
            self.stack.append(item)

    def pop(self):
        with self.lock:
            return self.stack.pop()

    def size(self):
        with self.lock:
            return len(self.stack)


"""
In this example, Stack is a thread-safe stack that can be used by multiple threads concurrently. The Lock instance self.lock is used to ensure that only one thread can execute the push, pop, and size methods at a time.

The Monitor Object pattern provides a simple and effective way to achieve thread safety in a multi-threaded environment. However, it can potentially introduce contention, which may lead to performance issues if not used carefully.
"""
