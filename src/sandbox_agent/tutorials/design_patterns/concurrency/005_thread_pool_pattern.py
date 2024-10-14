# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# pyright: ignore[reportCallInDefaultInitializer]
# pyright: ignore[reportAttributeAccessIssue]
"""
Thread Pool Pattern

Overview of the Thread Pool Pattern
The Thread Pool pattern is a concurrency design pattern that's used to manage a pool of worker threads. The threads in the pool are maintained ready to perform a certain kind of task. This is useful when there is a large number of short tasks to be performed rather than instantiating new threads. It's beneficial when there is a small number of long tasks, as the overhead of creating a large number of threads can impact performance and system stability.

Implementing the Thread Pool Pattern in Python
Python's standard library provides the concurrent.futures module, which offers a powerful, high-level interface for asynchronously executing callables, making it a perfect tool for implementing the Thread Pool pattern.

Here's an example of how to use it to calculate Fibonacci numbers:
"""

from __future__ import annotations

import concurrent.futures


def fib(n: int) -> int:
    """
    Calculate the nth Fibonacci number.

    Args:
        n (int): The index of the Fibonacci number to calculate.

    Returns:
        int: The nth Fibonacci number.
    """
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


def main() -> None:
    """Main function to demonstrate the Thread Pool pattern."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fib, n) for n in range(30)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()

"""
In this example, fib is a function that calculates a Fibonacci number. The ThreadPoolExecutor is a class that creates a pool of worker threads. executor.submit(fib, n) submits a callable to be executed and returns a Future object representing the execution of the callable. concurrent.futures.as_completed yields Futures as they complete.

The Thread Pool pattern allows us to control the number of threads, reuse threads, and have a better performance and system stability. However, it can lead to higher complexity due to the need to manage and coordinate threads and tasks.
"""
