"""
This type stub file was generated by pyright.
"""

import asyncio
import threading
from typing import Optional

PY_310 = ...
class AsyncQueue(asyncio.Queue):
    """Async unbounded FIFO queue with a wait() method.

    Subclassed from asyncio.Queue, adding a wait() method."""
    async def wait(self) -> None:
        """If queue is empty, wait until an item is available.

        Copied from Queue.get(), removing the call to .get_nowait(),
        ie. this doesn't consume the item, just waits for it.
        """
        ...



class Semaphore(threading.Semaphore):
    """Semaphore subclass with a wait() method."""
    def wait(self, blocking: bool = ..., timeout: Optional[float] = ...): # -> bool:
        """Block until the semaphore can be acquired, but don't acquire it."""
        ...



class SyncQueue:
    """Unbounded FIFO queue with a wait() method.
    Adapted from pure Python implementation of queue.SimpleQueue.
    """
    def __init__(self) -> None:
        ...

    def put(self, item, block=..., timeout=...): # -> None:
        """Put the item on the queue.

        The optional 'block' and 'timeout' arguments are ignored, as this method
        never blocks.  They are provided for compatibility with the Queue class.
        """
        ...

    def get(self, block=..., timeout=...):
        """Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until an item is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately
        available, else raise the Empty exception ('timeout' is ignored
        in that case).
        """
        ...

    def wait(self, block=..., timeout=...): # -> None:
        """If queue is empty, wait until an item maybe is available,
        but don't consume it.
        """
        ...

    def empty(self): # -> bool:
        """Return True if the queue is empty, False otherwise (not reliable!)."""
        ...

    def qsize(self): # -> int:
        """Return the approximate size of the queue (not reliable!)."""
        ...

    __class_getitem__ = ...


__all__ = ["AsyncQueue", "SyncQueue"]
