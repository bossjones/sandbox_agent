# pragma: exclude file
"""
Implementing a Database Connection Pool using the Object Pool Pattern

The Object Pool Pattern is a design pattern that uses a set of initialized objects, the "pool," ready to use rather than allocating and deallocating them on the fly. When an object is taken from the pool, it is not available until it is returned. Objects in the pool have a lifecycle: creation, validation, and destruction.

In this module, we implement a Database Connection Pool using the Object Pool design pattern. Connection pools are often used to enhance the performance of executing commands on a database. Opening and maintaining a database connection for each user, especially requests made to a dynamic database-driven website application, is costly and wastes resources.
"""

from __future__ import annotations

import sqlite3

from abc import ABC, abstractmethod
from typing import List


class ObjectPool(ABC):
    """
    Abstract base class for object pool implementations.
    """

    def __init__(self) -> None:
        """
        Initialize the object pool with an empty list of resources.
        """
        self._resources: list = []

    @abstractmethod
    def create_resource(self) -> object:
        """
        Abstract method to create a new resource.

        Returns:
            object: The created resource.
        """
        pass

    def acquire(self) -> object:
        """
        Acquire a resource from the pool or create a new one if the pool is empty.

        Returns:
            object: The acquired resource.
        """
        if len(self._resources) > 0:
            return self._resources.pop()
        else:
            return self.create_resource()

    def release(self, resource: object) -> None:
        """
        Release a resource back to the pool.

        Args:
            resource (object): The resource to be released.
        """
        self._resources.append(resource)


class DatabaseConnectionPool(ObjectPool):
    """
    Concrete implementation of an object pool for database connections.
    """

    def create_resource(self) -> sqlite3.Connection:
        """
        Create a new SQLite database connection.

        Returns:
            sqlite3.Connection: The created database connection.
        """
        return sqlite3.connect(":memory:")  # Creates an in-memory database for simplicity


def use_database(pool: DatabaseConnectionPool) -> None:
    """
    Use a database connection from the pool to perform some operations.

    Args:
        pool (DatabaseConnectionPool): The database connection pool.
    """
    connection = pool.acquire()
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS Test (id INTEGER)")
    cursor.execute("INSERT INTO Test (id) VALUES (1)")
    connection.commit()
    pool.release(connection)


if __name__ == "__main__":
    pool = DatabaseConnectionPool()
    use_database(pool)

"""
In this code:
- ObjectPool is the abstract base class that manages resources. It provides acquire() and release() methods to get and return resources to the pool. create_resource() is an abstract method that concrete pools need to implement.
- DatabaseConnectionPool is a subclass that provides a concrete implementation for create_resource(). It creates a new SQLite database connection.
- use_database() is a function that uses a database connection from the pool to perform some operations and then returns the connection to the pool.

This simple example illustrates the essence of the Object Pool pattern. A more sophisticated implementation might include features like a maximum pool size, resource validation, automatic resource cleanup, etc. However, in its simplest form, the Object Pool pattern provides an effective way to manage and reuse expensive resources like database connections.
"""
