from transformers.models.gemma.modeling_gemma import GemmaForSequenceClassification as GemmaSeqClassModel
from transformers.models.llama.configuration_llama import LlamaConfig


# Demonstration where modifications are limited to docstring alterations only
class MyNewModel2Config(LlamaConfig):
    r"""
    Configuration class for storing the setup of a [`GemmaModel`]. This class enables instantiation of an Gemma
    model based on given arguments, specifying the underlying model architecture. Creating a configuration with default
    settings produces a similar configuration to that of the Gemma-7B baseline.
    e.g. [google/gemma-7b](https://huggingface.co/google/gemma-7b)
    Configuration objects inherit from [`PreTrainedConfig`] and may be utilized to manage the model outputs. Refer to
    the documentation from [`PreTrainedConfig`] for additional details.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary dimension of the Gemma model. Specifies how many unique tokens can be handled by the
            `inputs_ids` when invoking [`GemmaModel`]