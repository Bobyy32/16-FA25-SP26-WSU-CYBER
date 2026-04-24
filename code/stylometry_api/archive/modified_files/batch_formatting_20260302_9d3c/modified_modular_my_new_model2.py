from transformers.models.gemma.modeling_gemma import GemmaForSequenceClassification
from transformers.models.llama.configuration_llama import LlamaConfig


# Example where we just want to only tweak the docstring content
class MyNewModel2Config(LlamaConfig):
    r"""
    This is the configuration class used to store the configuration of a [`GemmaModel`]. It enables instantiating an Gemma
    model according to specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma-7B.
    e.g. [google/gemma-7b](https://huggingface.co/google/gemma-7b)
    Configuration objects inherit from [`PreTrainedConfig`] and can control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more details.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma model. Defines how many different tokens that can be represented by the
            `inputs_ids` passed when calling [`GemmaModel`]