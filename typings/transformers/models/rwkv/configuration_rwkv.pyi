"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

""" RWKV configuration"""
logger = ...
class RwkvConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`RwkvModel`]. It is used to instantiate a RWKV
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RWVK-4
    [RWKV/rwkv-4-169m-pile](https://huggingface.co/RWKV/rwkv-4-169m-pile) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50277):
            Vocabulary size of the RWKV model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RwkvModel`].
        context_length (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model can be used with in a single forward (using it in RNN mode
            lets use any sequence length).
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        attention_hidden_size (`int`, *optional*):
            Dimensionality of the attention hidden states. Will default to `hidden_size` if unset.
        intermediate_size (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. Will default to 4 times `hidden_size` if unset.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer
            as GPTNeoX.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer as
            GPTNeoX.
        rescale_every (`int`, *optional*, defaults to 6):
            At inference, the hidden states (and weights of the correponding output layers) are divided by 2 every
            `rescale_every` layer. If set to 0 or a negative number, no rescale is done.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the input token embeddings.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last state.


    Example:

    ```python
    >>> from transformers import RwkvConfig, RwkvModel

    >>> # Initializing a Rwkv configuration
    >>> configuration = RwkvConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RwkvModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    attribute_map = ...
    def __init__(self, vocab_size=..., context_length=..., hidden_size=..., num_hidden_layers=..., attention_hidden_size=..., intermediate_size=..., layer_norm_epsilon=..., bos_token_id=..., eos_token_id=..., rescale_every=..., tie_word_embeddings=..., use_cache=..., **kwargs) -> None:
        ...

