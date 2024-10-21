# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# pyright: ignore[reportCallInDefaultInitializer]
# pyright: ignore[reportAttributeAccessIssue]
"""
Future-Promise Pattern

Overview of the Future-Promise Pattern

The Future-Promise pattern is a concurrency design pattern that is used to represent a computation that may not have been finished.
It's used as a proxy for a result that is initially unknown, often because the computation of its value isn't yet complete.

- The Future part of the pattern refers to a value or error that isn't necessarily known when the Future object is created. It is a read-only placeholder view of a variable.
- The Promise is a writable single assignment container which sets the value of the future. Essentially, the Promise is the producer of the value, and the Future is the consumer of the value.
"""

"""
Implementing the Future-Promise Pattern in Python

Python's standard library offers the concurrent.futures module for a high-level interface for asynchronously executing callables.
The asynchronous execution can be performed with threads, using ThreadPoolExecutor, or separate processes, using ProcessPoolExecutor.
"""
from __future__ import annotations

import concurrent.futures
import urllib.request


URLS = ["http://www.python.org", "http://www.python.org/about"]


def load_url(url: str, timeout: int) -> bytes:
    """
    Load a URL and return its contents.

    Args:
        url (str): The URL to load.
        timeout (int): The timeout in seconds.

    Returns:
        bytes: The contents of the URL.
    """
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    future_to_url: dict[concurrent.futures.Future[bytes], str] = {
        executor.submit(load_url, url, 60): url for url in URLS
    }

    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url!r} generated an exception: {exc}")
        else:
            print("%r page length is %d" % (url, len(data)))

"""
In this example, load_url is a function that takes a URL and returns its contents.
The concurrent.futures.ThreadPoolExecutor is a context manager that runs load_url in separate threads.
The executor.submit method schedules the callable, load_url, to be executed and returns a Future object representing the execution of the callable.

The Future-Promise pattern can simplify async programming by providing a convenient way to structure your program when it uses many computations
that depend on each other but can run independently or in partial orders. It's essential to handle exceptions properly, as an exception in a worker thread
won't necessarily terminate your program as it does in the main thread.
"""
