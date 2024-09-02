"""
This type stub file was generated by pyright.
"""

from abc import ABC, abstractmethod
from typing import Optional
from grpc._channel import Channel
from pinecone import Config
from .config import GRPCClientConfig

_logger = ...
class GRPCIndexBase(ABC):
    """
    Base class for grpc-based interaction with Pinecone indexes
    """
    _pool = ...
    def __init__(self, index_name: str, config: Config, channel: Optional[Channel] = ..., grpc_config: Optional[GRPCClientConfig] = ..., _endpoint_override: Optional[str] = ...) -> None:
        ...

    @property
    @abstractmethod
    def stub_class(self): # -> None:
        ...

    @property
    def channel(self): # -> grpc._channel.Channel | grpc.Channel:
        """Creates GRPC channel."""
        ...

    def grpc_server_on(self) -> bool:
        ...

    def close(self): # -> None:
        """Closes the connection to the index."""
        ...

    def __enter__(self): # -> Self:
        ...

    def __exit__(self, exc_type, exc_value, traceback): # -> None:
        ...

