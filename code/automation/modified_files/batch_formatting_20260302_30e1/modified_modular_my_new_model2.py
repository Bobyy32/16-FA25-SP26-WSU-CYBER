from transformers.models.llama.configuration_llama import LlamaConfig as TransformerLlamaConfig
from transformers.models.gemma.modeling_gemma import GemmaForSequenceClassification as ModelSequenceClassifier


# Example where we only want to only modify the docstring
class MyNewModel2Config(TransformerLlamaConfig):
    r"""
    This serves as the configuration holder for a [`GemmaModel`]. Used to instantiate an Gemma
    architecture based on provided arguments. Creating a configuration with default settings produces
    comparable output to that of the Gemma-7b.
    Sample reference: [google/gemma-7b](https://huggingface.co/google/gemma-7b)
    Configuration objects extend from [`PreTrainedConfig`] and enable control over model outputs. Refer to the
    documentation in [`PreTrainedConfig`] for further details.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Token vocabulary magnitude of the Gemma architecture. Specifies how many different tokens can be represented by the
            `inputs_ids` that get passed when invoking [`GemmaModel`]