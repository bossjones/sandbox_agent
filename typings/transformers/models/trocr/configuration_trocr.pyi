"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

""" TrOCR model configuration"""
logger = ...
class TrOCRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TrOCRForCausalLM`]. It is used to instantiate an
    TrOCR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the TrOCR
    [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the TrOCR model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TrOCRForCausalLM`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the pooler. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the word embeddings by sqrt(d_model).
        use_learned_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether or not to use learned position embeddings. If not, sinusoidal position embeddings will be used.
        layernorm_embedding (`bool`, *optional*, defaults to `True`):
            Whether or not to use a layernorm after the word + position embeddings.

    Example:

    ```python
    >>> from transformers import TrOCRConfig, TrOCRForCausalLM

    >>> # Initializing a TrOCR-base style configuration
    >>> configuration = TrOCRConfig()

    >>> # Initializing a model (with random weights) from the TrOCR-base style configuration
    >>> model = TrOCRForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(self, vocab_size=..., d_model=..., decoder_layers=..., decoder_attention_heads=..., decoder_ffn_dim=..., activation_function=..., max_position_embeddings=..., dropout=..., attention_dropout=..., activation_dropout=..., decoder_start_token_id=..., init_std=..., decoder_layerdrop=..., use_cache=..., scale_embedding=..., use_learned_position_embeddings=..., layernorm_embedding=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
        ...

