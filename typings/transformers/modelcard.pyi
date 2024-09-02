"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

""" Configuration base class and utilities."""
TASK_MAPPING = ...
logger = ...
class ModelCard:
    r"""
    Structured Model Card class. Store model card as well as methods for loading/downloading/saving model cards.

    Please read the following paper for details and explanation on the sections: "Model Cards for Model Reporting" by
    Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer,
    Inioluwa Deborah Raji and Timnit Gebru for the proposal behind model cards. Link: https://arxiv.org/abs/1810.03993

    Note: A model card can be loaded and saved to disk.
    """
    def __init__(self, **kwargs) -> None:
        ...

    def save_pretrained(self, save_directory_or_file): # -> None:
        """Save a model card object to the directory or file `save_directory_or_file`."""
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs): # -> tuple[Any | Self, dict[str, Any]] | Self:
        r"""
        Instantiate a [`ModelCard`] from a pre-trained model model card.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string, the *model id* of a pretrained model card hosted inside a model repo on huggingface.co.
                - a path to a *directory* containing a model card file saved using the [`~ModelCard.save_pretrained`]
                  method, e.g.: `./my_model_directory/`.
                - a path or url to a saved model card JSON *file*, e.g.: `./my_model_directory/modelcard.json`.

            cache_dir: (*optional*) string:
                Path to a directory in which a downloaded pre-trained model card should be cached if the standard cache
                should not be used.

            kwargs: (*optional*) dict: key/value pairs with which to update the ModelCard object after loading.

                - The values in kwargs of any keys which are model card attributes will be used to override the loaded
                  values.
                - Behavior concerning key/value pairs whose keys are *not* model card attributes is controlled by the
                  *return_unused_kwargs* keyword parameter.

            proxies: (*optional*) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}. The proxies are used on each request.

            return_unused_kwargs: (*optional*) bool:

                - If False, then this function returns just the final model card object.
                - If True, then this functions returns a tuple *(model card, unused_kwargs)* where *unused_kwargs* is a
                  dictionary consisting of the key/value pairs whose keys are not model card attributes: ie the part of
                  kwargs which has not been used to update *ModelCard* and is otherwise ignored.

        Examples:

        ```python
        # Download model card from huggingface.co and cache.
        modelcard = ModelCard.from_pretrained("google-bert/bert-base-uncased")
        # Model card was saved using *save_pretrained('./test/saved_model/')*
        modelcard = ModelCard.from_pretrained("./test/saved_model/")
        modelcard = ModelCard.from_pretrained("./test/saved_model/modelcard.json")
        modelcard = ModelCard.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        ```"""
        ...

    @classmethod
    def from_dict(cls, json_object): # -> Self:
        """Constructs a `ModelCard` from a Python dictionary of parameters."""
        ...

    @classmethod
    def from_json_file(cls, json_file): # -> Self:
        """Constructs a `ModelCard` from a json file of parameters."""
        ...

    def __eq__(self, other) -> bool:
        ...

    def __repr__(self): # -> str:
        ...

    def to_dict(self): # -> dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        ...

    def to_json_string(self): # -> str:
        """Serializes this instance to a JSON string."""
        ...

    def to_json_file(self, json_file_path): # -> None:
        """Save this instance to a json file."""
        ...



AUTOGENERATED_TRAINER_COMMENT = ...
AUTOGENERATED_KERAS_COMMENT = ...
TASK_TAG_TO_NAME_MAPPING = ...
METRIC_TAGS = ...
def infer_metric_tags_from_eval_results(eval_results): # -> dict[Any, Any]:
    ...

def is_hf_dataset(dataset): # -> bool:
    ...

@dataclass
class TrainingSummary:
    model_name: str
    language: Optional[Union[str, List[str]]] = ...
    license: Optional[str] = ...
    tags: Optional[Union[str, List[str]]] = ...
    finetuned_from: Optional[str] = ...
    tasks: Optional[Union[str, List[str]]] = ...
    dataset: Optional[Union[str, List[str]]] = ...
    dataset_tags: Optional[Union[str, List[str]]] = ...
    dataset_args: Optional[Union[str, List[str]]] = ...
    dataset_metadata: Optional[Dict[str, Any]] = ...
    eval_results: Optional[Dict[str, float]] = ...
    eval_lines: Optional[List[str]] = ...
    hyperparameters: Optional[Dict[str, Any]] = ...
    source: Optional[str] = ...
    def __post_init__(self): # -> None:
        ...

    def create_model_index(self, metric_mapping): # -> list[dict[str, str]]:
        ...

    def create_metadata(self): # -> dict[Any, Any]:
        ...

    def to_model_card(self): # -> str:
        ...

    @classmethod
    def from_trainer(cls, trainer, language=..., license=..., tags=..., model_name=..., finetuned_from=..., tasks=..., dataset_tags=..., dataset_metadata=..., dataset=..., dataset_args=...):
        ...

    @classmethod
    def from_keras(cls, model, model_name, keras_history=..., language=..., license=..., tags=..., finetuned_from=..., tasks=..., dataset_tags=..., dataset=..., dataset_args=...): # -> Self:
        ...



def parse_keras_history(logs): # -> tuple[None, list[Any], dict[Any, Any]] | tuple[Any | dict[Any, list[Any]], list[Any], Any]:
    """
    Parse the `logs` of either a `keras.History` object returned by `model.fit()` or an accumulated logs `dict`
    passed to the `PushToHubCallback`. Returns lines and logs compatible with those returned by `parse_log_history`.
    """
    ...

def parse_log_history(log_history): # -> tuple[None, None, Any] | tuple[None, None, None] | tuple[Any, list[Any], dict[Any, Any]] | tuple[Any, list[Any], None]:
    """
    Parse the `log_history` of a Trainer to get the intermediate and final evaluation results.
    """
    ...

def extract_hyperparameters_from_keras(model): # -> dict[Any, Any]:
    ...

def make_markdown_table(lines): # -> LiteralString | Literal['']:
    """
    Create a nice Markdown table from the results in `lines`.
    """
    ...

_TRAINING_ARGS_KEYS = ...
def extract_hyperparameters_from_trainer(trainer): # -> dict[str, Any]:
    ...
