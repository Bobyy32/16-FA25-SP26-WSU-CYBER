import transformers.models.gemma.modeling_gemma as gemma_modeling
from transformers.models.llama.configuration_llama import LlamaConfig


# Example where we only want to only modify the docstring
class MyNewModel2Config(LlamaConfig):
    r"""
    This serves as the configuration class for storing settings of a [`GemmaModel`]. It enables instantiating an Gemma
    model based on provided arguments, establishing the model architecture. Creating a configuration with
    default parameters will produce a configuration comparable to that found in Gemma-7B.
    Sample reference: [google/gemma-7b](https://huggingface.co/google/gemma-7b)
    Configuration objects extend [`PreTrainedConfig`] and enable control over model outputs. See the
    documentation from [`PreTrainedConfig`] for further details.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma model. Specifies the quantity of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GemmaModel`]