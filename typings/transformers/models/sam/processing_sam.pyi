"""
This type stub file was generated by pyright.
"""

from typing import Optional, Union
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_tf_available, is_torch_available

"""
Processor class for SAM.
"""
if is_torch_available():
    ...
if is_tf_available():
    ...
class SamProcessor(ProcessorMixin):
    r"""
    Constructs a SAM processor which wraps a SAM image processor and an 2D points & Bounding boxes processor into a
    single processor.

    [`SamProcessor`] offers all the functionalities of [`SamImageProcessor`]. See the docstring of
    [`~SamImageProcessor.__call__`] for more information.

    Args:
        image_processor (`SamImageProcessor`):
            An instance of [`SamImageProcessor`]. The image processor is a required input.
    """
    attributes = ...
    image_processor_class = ...
    def __init__(self, image_processor) -> None:
        ...

    def __call__(self, images=..., segmentation_maps=..., input_points=..., input_labels=..., input_boxes=..., return_tensors: Optional[Union[str, TensorType]] = ..., **kwargs) -> BatchEncoding:
        """
        This method uses [`SamImageProcessor.__call__`] method to prepare image(s) for the model. It also prepares 2D
        points and bounding boxes for the model if they are provided.
        """
        ...

    @property
    def model_input_names(self): # -> list[Any]:
        ...

    def post_process_masks(self, *args, **kwargs):
        ...

