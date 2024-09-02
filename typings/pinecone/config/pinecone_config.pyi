"""
This type stub file was generated by pyright.
"""

from typing import Dict, Optional
from .config import Config

logger = ...
DEFAULT_CONTROLLER_HOST = ...
class PineconeConfig:
   @staticmethod
   def build(api_key: Optional[str] = ..., host: Optional[str] = ..., additional_headers: Optional[Dict[str, str]] = ..., **kwargs) -> Config:
      ...

