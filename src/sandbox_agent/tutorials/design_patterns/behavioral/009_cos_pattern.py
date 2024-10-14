# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Chain of Responsibility Pattern

Overview of the Chain of Responsibility Pattern

The Chain of Responsibility pattern is a behavioral design pattern that lets you pass requests along a chain of handlers.
Upon receiving a request, each handler decides either to process the request or to pass it to the next handler in the chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Handler(ABC):
    """
    Abstract base class representing a handler in the chain of responsibility.
    """

    def __init__(self) -> None:
        """
        Initialize the handler with no next handler.
        """
        self._next_handler: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        """
        Set the next handler in the chain.

        Args:
            handler (Handler): The next handler to be set.

        Returns:
            Handler: The next handler.
        """
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle_request(self, request: str) -> Optional[str]:
        """
        Handle the request or pass it to the next handler.

        Args:
            request (str): The request to be handled.

        Returns:
            Optional[str]: The result of handling the request, if applicable.
        """
        pass

    def pass_to_next(self, request: str) -> Optional[str]:
        """
        Pass the request to the next handler, if available.

        Args:
            request (str): The request to be passed.

        Returns:
            Optional[str]: The result of handling the request by the next handler, if available.
        """
        if self._next_handler:
            return self._next_handler.handle_request(request)
        return None


class ConcreteHandler1(Handler):
    """
    Concrete handler class that handles specific requests.
    """

    def handle_request(self, request: str) -> Optional[str]:
        """
        Handle the request if it matches a specific condition, otherwise pass it to the next handler.

        Args:
            request (str): The request to be handled.

        Returns:
            Optional[str]: The result of handling the request, if applicable.
        """
        if request == "request1":
            return "Handler1 handling request1"
        else:
            return self.pass_to_next(request)


class ConcreteHandler2(Handler):
    """
    Concrete handler class that handles specific requests.
    """

    def handle_request(self, request: str) -> Optional[str]:
        """
        Handle the request if it matches a specific condition, otherwise pass it to the next handler.

        Args:
            request (str): The request to be handled.

        Returns:
            Optional[str]: The result of handling the request, if applicable.
        """
        if request == "request2":
            return "Handler2 handling request2"
        else:
            return self.pass_to_next(request)


# Client code
handler1 = ConcreteHandler1()
handler2 = ConcreteHandler2()

handler1.set_next(handler2)

# Send request1
print(handler1.handle_request("request1"))  # Outputs: Handler1 handling request1

# Send request2
print(handler1.handle_request("request2"))  # Outputs: Handler2 handling request2

"""
In this example, Handler is our Handler interface, and ConcreteHandler1 and ConcreteHandler2 are our Concrete Handlers.
When a request is made, ConcreteHandler1 tries to process it; if it can't, it passes the request to ConcreteHandler2.

The Chain of Responsibility pattern decouples sender and receiver of a request based on the type of request.
This pattern simplifies your code because the object that sends the request doesn't need to know the chain's structure and the receiver object.
However, it can lead to a performance hit as the request may need to be passed along the entire chain.
"""
