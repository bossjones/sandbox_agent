"""
Readers-Writers Pattern

Overview of the Readers-Writers Pattern
The Readers-Writers pattern is a concurrency design pattern that is used to solve one of the common concurrency problems. This problem occurs when a shared resource, typically a data structure or a database, is accessed concurrently by multiple threads or processes which might be reading data from the resource or writing data to it.

The pattern allows multiple readers to read at the same time as long as there are no writers writing. The write operations are exclusive, so when a writer is writing to the resource, all other writers or readers will be blocked until the writer is finished writing.

Implementing the Readers-Writers Pattern in Python
Python's standard library offers a threading module that can be used to implement this pattern. However, for the synchronization of access to the shared resource, we will need a bit more than the basic lock provided by the threading module. Here's a simple implementation using a custom ReadWriteLock class:
"""

from __future__ import annotations

import threading


class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()

    def release_write(self):
        self._read_ready.release()


# Usage
rwlock = ReadWriteLock()


# Reader
def reader():
    while True:
        rwlock.acquire_read()
        # Read data
        rwlock.release_read()


# Writer
def writer():
    while True:
        rwlock.acquire_write()
        # Write data
        rwlock.release_write()


"""
In this example, reader and writer are functions simulating the reading and writing operations. The ReadWriteLock class is the key element of the pattern, managing the access to the shared resource.

The Readers-Writers pattern can provide better performance than a mutual exclusion lock when the number of readers is much larger than the number of writers, but it might be a bit complex to implement due to the need for custom synchronization constructs.
"""
