from transformers import GemmaConfig
from transformers.models.gemma import GemmaForSequenceClassification
from transformers.models.llama import LlamaConfig


class MyNewModel2Config(LlamaConfig):
    """Configuration class for storing model parameters. Inherits from PreTrainedConfig to control outputs."""

    docstring_example = (
        "This is the configuration class to store the configuration of a Gemma model.\n"
        "It is used to instantiate an Gemma model according to the specified arguments.\n\n"
        "Args:\n"
        "    vocab_size: Vocabulary size of the Gemma model. Defaults to 256000.\n\n"
        "Usage example:\n"
        ">>> from transformers import GemmaModel, GemmaConfig\n"
        ">>> configuration = GemmaConfig()\n"
        ">>> model = GemmaModel(configuration)"
    )


class MyNewModel2ForSequenceClassification(GemmaForSequenceClassification):
    pass