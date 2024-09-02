"""
This type stub file was generated by pyright.
"""

from typing import Dict, Optional
from torch import Tensor, nn

class _SimpleSegmentationModel(nn.Module):
    __constants__ = ...
    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = ...) -> None:
        ...

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        ...

