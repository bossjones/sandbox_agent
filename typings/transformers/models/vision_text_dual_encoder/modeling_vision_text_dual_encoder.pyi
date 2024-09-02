"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ..clip.modeling_clip import CLIPOutput
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

""" PyTorch VisionTextDualEncoder model."""
logger = ...
_CONFIG_FOR_DOC = ...
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = ...
VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING = ...
VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING = ...
VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = ...
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    ...

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    ...

@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class VisionTextDualEncoderModel(PreTrainedModel):
    config_class = VisionTextDualEncoderConfig
    base_model_prefix = ...
    def __init__(self, config: Optional[VisionTextDualEncoderConfig] = ..., vision_model: Optional[PreTrainedModel] = ..., text_model: Optional[PreTrainedModel] = ...) -> None:
        ...

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids=..., attention_mask=..., position_ids=..., token_type_ids=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import VisionTextDualEncoderModel, AutoTokenizer

        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")

        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import VisionTextDualEncoderModel, AutoImageProcessor

        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        ...

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., return_loss: Optional[bool] = ..., token_type_ids: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import (
        ...     VisionTextDualEncoderModel,
        ...     VisionTextDualEncoderProcessor,
        ...     AutoImageProcessor,
        ...     AutoTokenizer,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        >>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
        >>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        ...     "google/vit-base-patch16-224", "google-bert/bert-base-uncased"
        ... )

        >>> # contrastive training
        >>> urls = [
        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
        ...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
        ... ]
        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True
        ... )
        >>> outputs = model(
        ...     input_ids=inputs.input_ids,
        ...     attention_mask=inputs.attention_mask,
        ...     pixel_values=inputs.pixel_values,
        ...     return_loss=True,
        ... )
        >>> loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score

        >>> # save and load from pretrained
        >>> model.save_pretrained("vit-bert")
        >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert")

        >>> # inference
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        ...

    @classmethod
    def from_pretrained(cls, *args, **kwargs): # -> tuple[Any | Self, dict[str, Any] | dict[str, Any | list[Any]] | Any] | Self:
        ...

    @classmethod
    def from_vision_text_pretrained(cls, vision_model_name_or_path: str = ..., text_model_name_or_path: str = ..., *model_args, **kwargs) -> PreTrainedModel:
        """
        Params:
            vision_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the vision model. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.

            text_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text model. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text configuration, use the prefix *text_* for each configuration parameter.
                - To update the vision configuration, use the prefix *vision_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import VisionTextDualEncoderModel

        >>> # initialize a model from pretrained ViT and BERT models. Note that the projection layers will be randomly initialized.
        >>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        ...     "google/vit-base-patch16-224", "google-bert/bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionTextDualEncoderModel.from_pretrained("./vit-bert")
        ```"""
        ...

