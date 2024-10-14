"""
Half-Sync/Half-Async Pattern

Overview of the Half-Sync/Half-Async Pattern
The Half-Sync/Half-Async pattern is a concurrency pattern that separates synchronous processing in a concurrent system, simplifying programming without degrading execution efficiency.
It introduces a queueing layer between the synchronous and asynchronous processing modules. The asynchronous layer can handle events and input/output operations, and the synchronous layer focuses on application-specific processing. This decoupling allows the complexities related to the synchronization of the asynchronous and synchronous layers to be confined to the queueing layer.

This pattern is quite beneficial when used in systems like event-driven servers, where input/output operations can be processed asynchronously while application-specific processing is done synchronously.

Implementing the Half-Sync/Half-Async Pattern in Python
Implementing the Half-Sync/Half-Async pattern in Python can be achieved using constructs provided by the concurrent.futures and queue modules. Below is a simple example of how to implement it:
"""

from __future__ import annotations

import concurrent.futures
import queue
import time


# Asynchronous layer
def producer(queue):
    for i in range(10):
        print(f"Producer is producing {i}")
        queue.put(i)
        time.sleep(0.5)


# Synchronous layer
def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"Consumer is consuming {item}")
        time.sleep(1)
        queue.task_done()


# Queueing layer
q = queue.Queue()

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(producer, q)
    executor.submit(consumer, q)

"""
In this example, producer is a function representing the asynchronous layer and puts them in the queue. consumer represents the synchronous layer, which consumes items from the queue. The queue serves as the queueing layer. The ThreadPoolExecutor is used to run the producer and consumer concurrently in separate threads.

The Half-Sync/Half-Async pattern allows us to separate concerns of asynchronous and synchronous processing, simplifying programming of concurrent systems. However, it also introduces a queueing layer, adding to the complexity of the system.
"""
