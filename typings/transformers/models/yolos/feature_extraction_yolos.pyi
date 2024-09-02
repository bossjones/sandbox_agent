"""
This type stub file was generated by pyright.
"""

from .image_processing_yolos import YolosImageProcessor

"""Feature extractor class for YOLOS."""
logger = ...
def rgb_to_id(x): # -> NDArray[signedinteger[Any]] | int:
    ...

class YolosFeatureExtractor(YolosImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        ...

