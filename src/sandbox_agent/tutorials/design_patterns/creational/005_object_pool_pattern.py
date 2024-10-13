# creational design patterns

"""
Object Pool Pattern

Overview of the Object Pool Pattern The Object Pool pattern is a software creational design pattern that uses a set of initialized objects kept ready to use, rather than allocating and destroying them on demand. A client of the pool will request an object from the pool and perform operations on the returned object. When the client has finished, it returns the object, which is a specific type of factory object, to the pool rather than destroying it. This can be more performant and memory-friendly in scenarios where the cost of initialization is high, the rate of instantiation of a class is high, and the number of instances in use at any one time is low. Implementing the Object Pool Pattern in Python Let's illustrate this pattern with an example, where we create a pool of string objects.

Envall, Henrik. Python Design Patterns: A Very Practical Guide (p. 19). Kindle Edition.
"""

from __future__ import annotations


class ObjectPool:
    """
    Manages a pool of reusable objects.
    """

    def __init__(self, max_objects: int = 10) -> None:
        """
        Initialize an ObjectPool instance with a maximum number of objects.

        Args:
            max_objects (int, optional): The maximum number of objects in the pool. Defaults to 10.
        """
        self._objects: list[str] = []
        self._max_objects = max_objects

    def get(self) -> str:
        """
        Get an object from the pool or create a new one if the pool is empty.

        Returns:
            str: An object from the pool or a newly created object.

        Raises:
            Exception: If the pool is empty and the maximum number of objects has been reached.
        """
        if not self._objects:
            if len(self._objects) < self._max_objects:
                self._objects.append(self._create())
            else:
                raise Exception("Could not create more objects.")
        return self._objects.pop()

    def return_object(self, obj: str) -> None:
        """
        Return an object to the pool.

        Args:
            obj (str): The object to return to the pool.
        """
        self._objects.append(obj)

    def _create(self) -> str:
        """
        Create a new object.

        Returns:
            str: A new empty string object.
        """
        return ""


string_pool = ObjectPool(10)
s1 = string_pool.get()
s2 = string_pool.get()
string_pool.return_object(s1)
string_pool.return_object(s2)

# In this code, ObjectPool manages a list of string objects. When get is called, it pops an object from the list if one is available, or creates a new one if the list is empty and the max limit has not been reached. The return_object method allows a client to return an object to the pool when it is done with it. This pattern can be especially beneficial in a multi-threaded application where many threads need to use a common resource, such as a database connection, and we want to limit the maximum number of connections. The Object Pool pattern can also improve performance by reusing objects and saving the overhead of initialization. However, it introduces complexity and potential resource contention, and can lead to increased memory use if objects in the pool are not reused effectively.
