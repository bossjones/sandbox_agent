"""
This type stub file was generated by pyright.
"""

from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import ArgumentHandler, Pipeline, build_pipeline_init_args

if is_torch_available():
    ...
if is_tf_available():
    ...
class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline
    """
    def __call__(self, table=..., query=..., **kwargs): # -> dict[Any, Any] | list[Any] | GeneratorType[Any, Any, Any] | list[dict[Any, Any]] | list[dict[str, Any]]:
        ...



@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class TableQuestionAnsweringPipeline(Pipeline):
    """
    Table Question Answering pipeline using a `ModelForTableQuestionAnswering`. This pipeline is only available in
    PyTorch.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
    >>> table = {
    ...     "Repository": ["Transformers", "Datasets", "Tokenizers"],
    ...     "Stars": ["36542", "4512", "3934"],
    ...     "Contributors": ["651", "77", "34"],
    ...     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
    ... }
    >>> oracle(query="How many stars does the transformers repository have?", table=table)
    {'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This tabular question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"table-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=table-question-answering).
    """
    default_input_names = ...
    def __init__(self, args_parser=..., *args, **kwargs) -> None:
        ...

    def batch_inference(self, **inputs): # -> Any:
        ...

    def sequential_inference(self, **inputs):
        """
        Inference used for models that need to process sequences in a sequential fashion, like the SQA models which
        handle conversational query related to a table.
        """
        ...

    def __call__(self, *args, **kwargs): # -> Any | Tensor | list[Any] | PipelineIterator | Generator[Any, Any, None] | None:
        r"""
        Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

        - `pipeline(table, query)`
        - `pipeline(table, [query])`
        - `pipeline(table=table, query=query)`
        - `pipeline(table=table, query=[query])`
        - `pipeline({"table": table, "query": query})`
        - `pipeline({"table": table, "query": [query]})`
        - `pipeline([{"table": table, "query": query}, {"table": table, "query": query}])`

        The `table` argument should be a dict or a DataFrame built from that dict, containing the whole table:

        Example:

        ```python
        data = {
            "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
            "age": ["56", "45", "59"],
            "number of movies": ["87", "53", "69"],
            "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
        }
        ```

        This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

        Example:

        ```python
        import pandas as pd

        table = pd.DataFrame.from_dict(data)
        ```

        Args:
            table (`pd.DataFrame` or `Dict`):
                Pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values.
                See above for an example of dictionary.
            query (`str` or `List[str]`):
                Query or list of queries that will be sent to the model alongside the table.
            sequential (`bool`, *optional*, defaults to `False`):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).

            truncation (`bool`, `str` or [`TapasTruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument `max_length`
                  or to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate row by row, removing rows from the table.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).


        Return:
            A dictionary or a list of dictionaries containing results: Each result is a dictionary with the following
            keys:

            - **answer** (`str`) -- The answer of the query given the table. If there is an aggregator, the answer will
              be preceded by `AGGREGATOR >`.
            - **coordinates** (`List[Tuple[int, int]]`) -- Coordinates of the cells of the answers.
            - **cells** (`List[str]`) -- List of strings made up of the answer cell values.
            - **aggregator** (`str`) -- If the model has an aggregator, this returns the aggregator.
        """
        ...

    def preprocess(self, pipeline_input, sequential=..., padding=..., truncation=...): # -> BatchEncoding:
        ...

    def postprocess(self, model_outputs): # -> list[dict[str, str]] | dict[str, str]:
        ...

