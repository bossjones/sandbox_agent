"""
This type stub file was generated by pyright.
"""

from typing import Dict, Tuple, Union
from pinecone.core.client.models import Vector

class VectorFactory:
    @staticmethod
    def build(item: Union[Vector, Tuple, Dict], check_type: bool = ...) -> Vector:
        ...

