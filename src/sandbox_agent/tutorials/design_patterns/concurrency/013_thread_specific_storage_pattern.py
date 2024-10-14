# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# pyright: ignore[reportCallInDefaultInitializer]
# pyright: ignore[reportAttributeAccessIssue]

"""
Thread-Specific Storage Pattern

Overview of the Thread-Specific Storage Pattern
The Thread-Specific Storage pattern allows multiple threads to use one logically shared object to retrieve an object that is local to a thread, without incurring locking overhead on each object access. This design pattern gives more object-oriented control over thread-local storage, which helps you write cleaner and more modular code.

Implementing the Thread-Specific Storage Pattern in Python
Python's threading module includes a local() function that creates an object capable of supporting thread-local data, thus allowing you to implement the Thread-Specific Storage pattern. Here is an example:
"""

from __future__ import annotations

import threading


# Thread-local data storage
local_storage = threading.local()


def display_storage(name):
    try:
        val = local_storage.val
    except AttributeError:
        print(f"{name}: No value yet")
    else:
        print(f"{name}: value = {val}")


def worker(number):
    print(f"Worker: {number}")
    if number < 3:
        local_storage.val = number  # Store thread-specific value
    display_storage("Worker")


# Create and start threads
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

"""
In this example, local_storage is an object that will hold data that is local to each thread. The worker function is run in each thread and sets a val attribute on local_storage if the worker's number is less than 3.

Remember that although this pattern can significantly simplify your code by avoiding unnecessary synchronization, it can also lead to issues if not managed correctly. For instance, it can increase memory consumption if large amounts of data are stored in thread-local storage. As always, use this pattern judiciously.
"""
