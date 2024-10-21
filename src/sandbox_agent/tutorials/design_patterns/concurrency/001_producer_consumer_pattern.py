# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# pyright: ignore[reportCallInDefaultInitializer]
# pyright: ignore[reportAttributeAccessIssue]

"""
CONCURRENCY PATTERNS

Concurrency patterns are integral in creating efficient, scalable, and robust applications, especially in a multithreaded or distributed environment.
Python provides several features and modules for concurrent programming such as threads, asyncio, and multiprocessing.
"""

"""
Producer-Consumer Pattern

Overview of the Producer-Consumer Pattern

The Producer-Consumer pattern is a concurrency design pattern that aims to parallelize data processing in your software.
In this pattern, there are two main actors: the Producer and the Consumer.
The Producer's job is to generate data, put it into a buffer, and start again. At the same time, the Consumer is consuming the data
(i.e., removing it from the buffer) one piece at a time.
"""

"""
Implementing the Producer-Consumer Pattern in Python

Python's standard library offers a great module for implementing this pattern: queue.
It provides the necessary locking semantics to handle multiple producers and multiple consumers.

Let's consider an example where the producers are generating numbers and the consumer is printing them:
"""

from __future__ import annotations

import queue
import random
import threading


class Producer(threading.Thread):
    """Producer class that generates data and puts it into a queue."""

    def __init__(self, queue: queue.Queue):
        """
        Initialize the Producer.

        Args:
            queue (queue.Queue): The queue to put generated data into.
        """
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self) -> None:
        """Run the producer thread."""
        for _ in range(10):  # Producer will produce 10 items
            item = random.randint(0, 256)  # Produce a random item
            self.queue.put(item)
            print(f"Producer notify: item {item} appended to queue by {self.name}\n")
            threading.Event().wait(1)


class Consumer(threading.Thread):
    """Consumer class that consumes data from a queue."""

    def __init__(self, queue: queue.Queue):
        """
        Initialize the Consumer.

        Args:
            queue (queue.Queue): The queue to consume data from.
        """
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self) -> None:
        """Run the consumer thread."""
        while True:
            item = self.queue.get()  # Consume an item
            print(f"Consumer notify: {item} popped from list by {self.name}\n")
            self.queue.task_done()


# Client code
q: queue.Queue = queue.Queue()
producer = Producer(q)
consumer = Consumer(q)

producer.start()
consumer.start()

producer.join()
consumer.join()

"""
In this example, Producer is our producer class that produces data and puts it into the queue, and Consumer is our consumer class that consumes data from the queue.

The Producer-Consumer pattern can help to decouple your components and enhance parallelism in your software, leading to potential performance improvements.
However, care must be taken to handle synchronization and race conditions between producers and consumers.
This is handled by the queue.Queue class in Python, which ensures thread safety.
"""
