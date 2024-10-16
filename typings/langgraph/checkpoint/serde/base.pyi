"""
This type stub file was generated by pyright.
"""

from typing import Any, Protocol

class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects.

    - `dumps`: Serialize an object to bytes.
    - `dumps_typed`: Serialize an object to a tuple (type, bytes).
    - `loads`: Deserialize an object from bytes.
    - `loads_typed`: Deserialize an object from a tuple (type, bytes).

    Valid implementations include the `pickle`, `json` and `orjson` modules.
    """
    def dumps(self, obj: Any) -> bytes:
        ...

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        ...

    def loads(self, data: bytes) -> Any:
        ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        ...



class SerializerCompat(SerializerProtocol):
    def __init__(self, serde: SerializerProtocol) -> None:
        ...

    def dumps(self, obj: Any) -> bytes:
        ...

    def loads(self, data: bytes) -> Any:
        ...

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        ...



def maybe_add_typed_methods(serde: SerializerProtocol) -> SerializerProtocol:
    """Wrap serde old serde implementations in a class with loads_typed and dumps_typed for backwards compatibility."""
    ...
