# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# pyright: ignore[reportCallInDefaultInitializer]
# pyright: ignore[reportAttributeAccessIssue]
"""
Scheduler Pattern

Overview of the Scheduler Pattern
The Scheduler pattern is a design pattern that helps in managing tasks and the resources required to run them. The main purpose of this pattern is to allow client applications to execute commands or perform tasks at different times based on certain criteria.

The Scheduler pattern is generally used in situations where tasks must be scheduled periodically, need to be performed at a specific time, or need to be prioritized based on certain rules.

Implementing the Scheduler Pattern in Python
Python provides the built-in sched module that can be used to implement a scheduler. Here's an example of how you might use it:
"""

from __future__ import annotations

import sched
import time


# Create a new scheduler
s = sched.scheduler(time.time, time.sleep)


def print_time() -> None:
    """Print the current time."""
    print("Current Time:", time.time())


def print_hello() -> None:
    """Print 'Hello, world!'."""
    print("Hello, world!")


# Schedule events
s.enter(5, 1, print_time)  # After 5 seconds, priority 1
s.enter(10, 2, print_hello)  # After 10 seconds, priority 2

# Run the scheduler
s.run()

"""
In this example, sched.scheduler is used to create a new scheduler. s.enter is used to schedule a new event. The run method is used to start the scheduler.

This basic example shows how to use the built-in scheduling capabilities of Python. If more advanced scheduling needs, there are third-party libraries available, such as APScheduler.

The Scheduler pattern allows us to simplify the management of tasks execution, especially when they must be performed periodically or at specific times. However, the complexity of scheduling logic can increase with the complexity of the task dependencies and scheduling criteria.
"""
