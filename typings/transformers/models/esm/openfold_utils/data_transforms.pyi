"""
This type stub file was generated by pyright.
"""

import numpy as np
import torch
from typing import Dict

def make_atom14_masks(protein: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Construct denser atom positions (14 dimensions instead of 37)."""
    ...

def make_atom14_masks_np(batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    ...
