from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.gemma.modeling_gemma import GemmaForSequenceClassification


class MyNewModel2Config(LlamaConfig):
    """This is the configuration class to store the configuration of a GemmaModel. It is used to instantiate an Gemma model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Gemma-7B. Example: google/gemma-7b. Configuration objects inherit from PreTrainedConfig and can be used to control the model outputs. Read the documentation from PreTrainedConfig for more information.
    Args:
        vocab_size (int, optional, defaults to 256000): Vocabulary size of the Gemma model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling GemmaModel
    Example:
        >>> from transformers import GemmaModel, GemmaConfig
        >>> configuration = GemmaConfig()
        >>> model = GemmaModel(configuration)
        >>> configuration = model.config"""


class MyNewModel2ForSequenceClassification(GemmaForSequenceClassification):
    pass