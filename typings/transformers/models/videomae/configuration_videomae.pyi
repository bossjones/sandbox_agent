"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

""" VideoMAE model configuration"""
logger = ...
class VideoMAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VideoMAEModel`]. It is used to instantiate a
    VideoMAE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VideoMAE
    [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_frames (`int`, *optional*, defaults to 16):
            The number of frames in each video.
        tubelet_size (`int`, *optional*, defaults to 2):
            The number of tubelets.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to mean pool the final hidden states instead of using the final hidden state of the [CLS] token.
        decoder_num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        norm_pix_loss (`bool`, *optional*, defaults to `True`):
            Whether to normalize the target patch pixels.

    Example:

    ```python
    >>> from transformers import VideoMAEConfig, VideoMAEModel

    >>> # Initializing a VideoMAE videomae-base style configuration
    >>> configuration = VideoMAEConfig()

    >>> # Randomly initializing a model from the configuration
    >>> model = VideoMAEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, image_size=..., patch_size=..., num_channels=..., num_frames=..., tubelet_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., initializer_range=..., layer_norm_eps=..., qkv_bias=..., use_mean_pooling=..., decoder_num_attention_heads=..., decoder_hidden_size=..., decoder_num_hidden_layers=..., decoder_intermediate_size=..., norm_pix_loss=..., **kwargs) -> None:
        ...

