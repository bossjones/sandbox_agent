# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# pyright: ignore[reportCallInDefaultInitializer]
# pyright: ignore[reportAttributeAccessIssue]
"""
Leader-Follower Pattern

Overview of the Leader-Follower Pattern
The Leader-Follower pattern is a concurrency design model where multiple threads can efficiently demultiplex events and synchronously process requests. This pattern works with a set of worker threads where one thread, the leader, waits for events. When an event occurs, the leader thread begins processing the event, and another thread becomes the new leader who waits for the next event. This pattern can significantly improve throughput, latency, and scalability when handling service requests across multiple I/O channels.

Implementing the Leader-Follower Pattern in Python
Implementing the Leader-Follower pattern in Python requires advanced primitives, such as condition variables, provided by the threading module. Here is a basic example:
"""

from __future__ import annotations

import queue
import random
import threading
import time


class WorkerThread(threading.Thread):
    """A worker thread that processes events from a queue."""

    def __init__(self, name: str, event_queue: queue.Queue):
        """
        Initialize the worker thread.

        Args:
            name (str): The name of the worker thread.
            event_queue (queue.Queue): The queue to get events from.
        """
        threading.Thread.__init__(self)
        self.name = name
        self.event_queue = event_queue

    def run(self) -> None:
        """Run the worker thread."""
        while True:
            event = self.event_queue.get()  # Become the leader
            print(f"{self.name} is processing event: {event}")
            time.sleep(random.random())  # Simulate work
            self.event_queue.task_done()  # Allow a new leader to be chosen


def main() -> None:
    """Main function to demonstrate the Leader-Follower pattern."""
    # Create a queue and some worker threads
    event_queue: queue.Queue = queue.Queue()
    for i in range(5):
        worker = WorkerThread(f"Worker-{i}", event_queue)
        worker.start()

    # Generate events
    for i in range(10):
        event_queue.put(f"Event-{i}")
        time.sleep(random.random())

    # Wait for all events to be processed
    event_queue.join()


if __name__ == "__main__":
    main()

"""
In this example, the WorkerThread class represents a worker thread that pulls events from the event_queue. When a worker thread pulls an event from the event_queue, it becomes the leader and starts processing the event. When it's done processing the event, it calls event_queue.task_done(), allowing a new leader to be chosen.

The Leader-Follower pattern allows us to develop efficient multithreaded servers. However, it can lead to higher complexity due to the need to carefully synchronize the leader election process to prevent race conditions.
"""
