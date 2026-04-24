from transformers.models.gemma.modeling_gemma import GemmaForSequenceClassification
from .transformers.models.llama.configuration_llama import LlamaConfig


class MyNewModel2Config(LlamaConfig):
    """
    Configuration class for storing model parameters of a [`GemmaModel`]. 
    Used to instantiate an Gemma
    model with specified arguments and defining architecture. 
    Defaults yield configuration similar to Gemma-7b.
    
    Example: [google/gemma-7b](https://huggingface.co/google/gemma-7b)
    Inherits from [`PreTrainedConfig`] for controlling model outputs.
    
    Args:
        vocab_size (int, optional, default=256000): Vocabulary size of the Gemma model defining number of different tokens that can be represented by inputs_ids passed to [`GemmaModel`]
    """


class MyNewModel2ForSequenceClassification(GemmaForSequenceClassification):
    """Extended sequence classification model using Gemma architecture"""
    pass