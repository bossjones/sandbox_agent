"""
This type stub file was generated by pyright.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions
from ...modeling_flax_utils import FlaxPreTrainedModel, add_start_docstrings_to_model_forward
from ...utils import add_start_docstrings, replace_return_docstrings
from .configuration_pegasus import PegasusConfig

""" Flax PEGASUS model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
PEGASUS_START_DOCSTRING = ...
PEGASUS_INPUTS_DOCSTRING = ...
PEGASUS_ENCODE_INPUTS_DOCSTRING = ...
PEGASUS_DECODE_INPUTS_DOCSTRING = ...
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    ...

def create_sinusoidal_positions(n_pos, dim):
    ...

class FlaxPegasusAttention(nn.Module):
    config: PegasusConfig
    embed_dim: int
    num_heads: int
    dropout: float = ...
    causal: bool = ...
    bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...

    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray] = ..., attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        ...



class FlaxPegasusEncoderLayer(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...



class FlaxPegasusEncoderLayerCollection(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[()], ...] | FlaxBaseModelOutput:
        ...



class FlaxPegasusDecoderLayer(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = ...
    def setup(self) -> None:
        ...

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., deterministic: bool = ...) -> Tuple[jnp.ndarray]:
        ...



class FlaxPegasusDecoderLayerCollection(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, hidden_states, attention_mask, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...



class FlaxPegasusEncoder(nn.Module):
    config: PegasusConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutput:
        ...



class FlaxPegasusDecoder(nn.Module):
    config: PegasusConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, position_ids, encoder_hidden_states: Optional[jnp.ndarray] = ..., encoder_attention_mask: Optional[jnp.ndarray] = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> tuple[Any, ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...



class FlaxPegasusModule(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> FlaxSeq2SeqModelOutput:
        ...



class FlaxPegasusPreTrainedModel(FlaxPreTrainedModel):
    config_class = PegasusConfig
    base_model_prefix: str = ...
    module_class: nn.Module = ...
    def __init__(self, config: PegasusConfig, input_shape: Tuple[int] = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., **kwargs) -> None:
        ...

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...

    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        ...

    @add_start_docstrings(PEGASUS_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=PegasusConfig)
    def encode(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

        >>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        ...

    @add_start_docstrings(PEGASUS_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=PegasusConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example:

        ```python
        >>> import jax.numpy as jnp
        >>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

        >>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        ...

    @add_start_docstrings_to_model_forward(PEGASUS_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., decoder_input_ids: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        ...



@add_start_docstrings("The bare Pegasus Model transformer outputting raw hidden-states without any specific head on top.", PEGASUS_START_DOCSTRING)
class FlaxPegasusModel(FlaxPegasusPreTrainedModel):
    config: PegasusConfig
    dtype: jnp.dtype = ...
    module_class = ...


class FlaxPegasusForConditionalGenerationModule(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., jnp.ndarray] = ...
    def setup(self): # -> None:
        ...

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ...): # -> Any | FlaxSeq2SeqLMOutput:
        ...



@add_start_docstrings("The PEGASUS Model with a language modeling head. Can be used for summarization.", PEGASUS_START_DOCSTRING)
class FlaxPegasusForConditionalGeneration(FlaxPegasusPreTrainedModel):
    module_class = ...
    dtype: jnp.dtype = ...
    @add_start_docstrings(PEGASUS_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=PegasusConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_position_ids: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., deterministic: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...): # -> FlaxCausalLMOutputWithCrossAttentions | Any:
        r"""
        Returns:

        Example:

        ```python
        >>> import jax.numpy as jnp
        >>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

        >>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""
        ...

    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, attention_mask: Optional[jax.Array] = ..., decoder_attention_mask: Optional[jax.Array] = ..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...



FLAX_PEGASUS_CONDITIONAL_GENERATION_DOCSTRING = ...
