"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...utils import is_flax_available, is_torch_available, is_vision_available

_import_structure = ...
if not is_vision_available():
    ...
if not is_torch_available():
    ...
if not is_flax_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
