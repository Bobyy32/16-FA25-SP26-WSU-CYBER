#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tuning the library models for permutation language modeling.
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    XLNetConfig,
    XLNetLMHeadModel,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Setup logging configuration for the training process
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///


def parse_configuration():
    """
    Parse command line arguments for training configuration.
    
    See all possible arguments in src/transformers/training_args.py
    or by passing the --help flag to this script.
    We now keep distinct sets of args, for a cleaner separation of concerns.
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_arguments, dataset_training_params, train_configuration = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_arguments, dataset_training_params, train_configuration = parser.parse_args_into_dataclasses()
    return model_arguments, dataset_training_params, train_configuration


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    
    --model_name_or_path: The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.
    --config_name: Pretrained config name or path if not the same as model_name
    --config_overrides: Override some existing default config settings when a model is trained from scratch
    --tokenizer_name: Pretrained tokenizer name or path if not the same as model_name
    --cache_dir: Where do you want to store the pretrained models downloaded from huggingface.co
    --use_fast_tokenizer: Whether to use one of the fast tokenizer (backed by the tokenizers library) or not
    --model_revision: The specific model version to use (can be a branch name, tag name or commit id)
    --token: The token to use as HTTP bearer authorization for remote files
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    
    --dataset_name: The name of the dataset to use (via the datasets library)
    --dataset_config_name: The configuration name of the dataset to use
    --trust_remote_code: Whether to trust the execution of code from datasets/models defined on the Hub
    --train_file: The input training data file (a text file)
    --validation_file: An optional input evaluation data file to evaluate the perplexity on
    --overwrite_cache: Overwrite the cached training and evaluation sets
    --validation_split_percentage: The percentage of the train set used as validation set
    --max_seq_length: The maximum total input sequence length after tokenization
    --preprocessing_num_workers: The number of processes to use for the preprocessing
    --plm_probability: Ratio of length of a span of masked tokens to surrounding context length
    --max_span_length: Maximum length of a span of masked tokens for permutation language modeling
    --line_by_line: Whether distinct lines of text in the dataset are to be handled as distinct sequences
    --pad_to_max_length: Whether to pad all samples to max_seq_length
    --max_train_samples: For debugging purposes or quicker training, truncate training examples
    --max_eval_samples: For debugging purposes or quicker training, truncate evaluation examples
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": (
                "Ratio of length of a span of masked tokens to surrounding context length for "
                "permutation language modeling."
            )
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def configure_training_environment(model_arguments, dataset_training_params, train_configuration):
    """
    Configure logging and set up the training environment.
    
    Set seed before initializing model.
    Log on each process the small summary of configuration details.
    Enable default logging handlers and explicit format.
    """
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    if train_configuration.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = train_configuration.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_configuration.local_process_index}, device: {train_configuration.device}, n_gpu: {train_configuration.n_gpu}, "
        + f"distributed training: {train_configuration.parallel_mode.value == 'distributed'}, 16-bits training: {train_configuration.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_configuration}")

    return model_arguments, dataset_training_params, train_configuration


def load_dataset_collection(dataset_training_params, model_arguments):
    """
    Load and prepare datasets for training.
    
    Downloading and loading a dataset from the hub.
    In distributed training, the load_dataset function guarantee that only one local process can concurrently
    download the dataset.
    """
    if dataset_training_params.dataset_name is not None:
        raw_datasets = load_dataset(
            dataset_training_params.dataset_name,
            dataset_training_params.dataset_config_name,
            cache_dir=model_arguments.cache_dir,
            token=model_arguments.token,
            trust_remote_code=dataset_training_params.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                dataset_training_params.dataset_name,
                dataset_training_params.dataset_config_name,
                split=f"train[:{dataset_training_params.validation_split_percentage}%]",
                cache_dir=model_arguments.cache_dir,
                token=model_arguments.token,
                trust_remote_code=dataset_training_params.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                dataset_training_params.dataset_name,
                dataset_training_params.dataset_config_name,
                split=f"train[{dataset_training_params.validation_split_percentage}%:]",
                cache_dir=model_arguments.cache_dir,
                token=model_arguments.token,
                trust_remote_code=dataset_training_params.trust_remote_code,
            )
    else:
        data_files = {}
        if dataset_training_params.train_file is not None:
            data_files["train"] = dataset_training_params.train_file
            extension = dataset_training_params.train_file.split(".")[-1]
        if dataset_training_params.validation_file is not None:
            data_files["validation"] = dataset_training_params.validation_file
            extension = dataset_training_params.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_arguments.cache_dir)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{dataset_training_params.validation_split_percentage}%]",
                cache_dir=model_arguments.cache_dir,
                token=model_arguments.token,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{dataset_training_params.validation_split_percentage}%:]",
                cache_dir=model_arguments.cache_dir,
                token=model_arguments.token,
            )

    return raw_datasets


def initialize_model_configuration(model_arguments):
    """
    Initialize model architecture and tokenizer configuration.
    
    Distributed training:
    The .from_pretrained methods guarantee that only one local process can concurrently
    download model & vocab.
    """
    config_kwargs = {
        "cache_dir": model_arguments.cache_dir,
        "revision": model_arguments.model_revision,
        "token": model_arguments.token,
    }
    if model_arguments.config_name:
        config_model_config = AutoConfig.from_pretrained(model_arguments.config_name, **config_kwargs)
    elif model_arguments.model_name_or_path:
        config_model_config = AutoConfig.from_pretrained(model_arguments.model_name_or_path, **config_kwargs)
    else:
        config_model_config = XLNetConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_arguments.config_overrides is not None:
            logger.info(f"Overriding config: {model_arguments.config_overrides}")
            config_model_config.update_from_string(model_arguments.config_overrides)
            logger.info(f"New config: {config_model_config}")

    tokenizer_kwargs = {
        "cache_dir": model_arguments.cache_dir,
        "use_fast": model_arguments.use_fast_tokenizer,
        "revision": model_arguments.model_revision,
        "token": model_arguments.token,
    }
    if model_arguments.tokenizer_name:
        tokenization_handler = AutoTokenizer.from_pretrained(model_arguments.tokenizer_name, **tokenizer_kwargs)
    elif model_arguments.model_name_or_path:
        tokenization_handler = AutoTokenizer.from_pretrained(model_arguments.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_arguments.model_name_or_path:
        language_model = XLNetLMHeadModel.from_pretrained(
            model_arguments.model_name_or_path,
            from_tf=bool(".ckpt" in model_arguments.model_name_or_path),
            config=config_model_config,
            cache_dir=model_arguments.cache_dir,
            revision=model_arguments.model_revision,
            token=model_arguments.token,
        )
    else:
        logger.info("Training new model from scratch")
        language_model = XLNetLMHeadModel(config_model_config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = language_model.get_input_embeddings().weight.shape[0]
    if len(tokenization_handler) > embedding_size:
        language_model.resize_token_embeddings(len(tokenization_handler))

    return config_model_config, tokenization_handler, language_model


def prepare_datasets_for_training(raw_datasets, dataset_training_params, tokenizer):
    """
    Preprocess and tokenize datasets for training.
    
    First we tokenize all the texts.
    Main data processing function that will concatenate all texts from our dataset and generate chunks of
    max_seq_length.
    """
    if train_configuration.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if dataset_training_params.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({dataset_training_params.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(dataset_training_params.max_seq_length, tokenizer.model_max_length)

    if dataset_training_params.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if dataset_training_params.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(examples["text"], padding=padding, truncation=True, max_length=max_seq_length)

        with train_configuration.main_process_first(desc="dataset map tokenization"):
            processed_datasets_collection = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=dataset_training_params.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not dataset_training_params.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with train_configuration.main_process_first(desc="dataset map tokenization"):
            processed_datasets_collection = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=dataset_training_params.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not dataset_training_params.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.

        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with train_configuration.main_process_first(desc="grouping texts together"):
            processed_datasets_collection = processed_datasets_collection.map(
                group_texts,
                batched=True,
                num_proc=dataset_training_params.preprocessing_num_workers,
                load_from_cache_file=not dataset_training_params.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    return processed_datasets_collection


def configure_training_pipeline(processed_datasets_collection, tokenizer, train_configuration):
    """
    Configure the training pipeline with collator and trainer.
    
    Data collator
    Initialize our Trainer
    """
    data_collation_strategy = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer,
        plm_probability=dataset_training_params.plm_probability,
        max_span_length=dataset_training_params.max_span_length,
    )

    # Training
    if train_configuration.do_train:
        if "train" not in processed_datasets_collection:
            raise ValueError("--do_train requires a train dataset")
        training_dataset = processed_datasets_collection["train"]
        if dataset_training_params.max_train_samples is not None:
            max_train_samples = min(len(training_dataset), dataset_training_params.max_train_samples)
            training_dataset = training_dataset.select(range(max_train_samples))

    # Evaluation
    if train_configuration.do_eval:
        if "validation" not in processed_datasets_collection:
            raise ValueError("--do_eval requires a validation dataset")
        evaluation_dataset = processed_datasets_collection["validation"]
        if dataset_training_params.max_eval_samples is not None:
            max_eval_samples = min(len(evaluation_dataset), dataset_training_params.max_eval_samples)
            evaluation_dataset = evaluation_dataset.select(range(max_eval_samples))

    return data_collation_strategy, training_dataset, evaluation_dataset


def orchestrate_training_pipeline(config_model_config, tokenization_handler, language_model, 
                                 data_collation_strategy, training_dataset, evaluation_dataset, 
                                 train_configuration):
    """
    Initialize and run the training trainer.
    
    Training loop and evaluation metrics handling.
    """
    # Initialize our Trainer
    training_automator = Trainer(
        model=language_model,
        args=train_configuration,
        train_dataset=training_dataset if train_configuration.do_train else None,
        eval_dataset=evaluation_dataset if train_configuration.do_eval else None,
        processing_classtokenizer=tokenization_handler,
        data_collator=data_collation_strategy,
    )

    # Training loop
    if train_configuration.do_train:
        checkpoint = None
        if train_configuration.resume_from_checkpoint is not None:
            checkpoint = train_configuration.resume_from_checkpoint
        train_result = training_automator.train(resume_from_checkpoint=checkpoint)
        training_automator.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            dataset_training_params.max_train_samples if dataset_training_params.max_train_samples is not None else len(training_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(training_dataset))

        training_automator.log_metrics("train", metrics)
        training_automator.save_metrics("train", metrics)
        training_automator.save_state()

    # Evaluation loop
    if train_configuration.do_eval:
        logger.info("*** Evaluate ***")

        metrics = training_automator.evaluate()

        max_eval_samples = dataset_training_params.max_eval_samples if dataset_training_params.max_eval_samples is not None else len(evaluation_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(evaluation_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        training_automator.log_metrics("eval", metrics)
        training_automator.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_arguments.model_name_or_path, "tasks": "language-modeling"}
    if dataset_training_params.dataset_name is not None:
        kwargs["dataset_tags"] = dataset_training_params.dataset_name
        if dataset_training_params.dataset_config_name is not None:
            kwargs["dataset_args"] = dataset_training_params.dataset_config_name
            kwargs["dataset"] = f"{dataset_training_params.dataset_name} {dataset_training_params.dataset_config_name}"
        else:
            kwargs["dataset"] = dataset_training_params.dataset_name

    if train_configuration.push_to_hub:
        training_automator.push_to_hub(**kwargs)
    else:
        training_automator.create_model_card(**kwargs)

    return metrics


def run_training_process(model_arguments, data_arguments, training_arguments):
    """
    Main entry point for the training process.
    
    Orchestrate the complete training pipeline from dataset loading to model evaluation.
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_arguments.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_arguments.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_arguments.local_process_index}, device: {training_arguments.device}, n_gpu: {training_arguments.n_gpu}, "
        + f"distributed training: {training_arguments.parallel_mode.value == 'distributed'}, 16-bits training: {training_arguments.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_arguments}")

    # Set seed before initializing model.
    set_seed(training_arguments.seed)

    # Load and prepare datasets
    raw_datasets = load_dataset(
        data_arguments.dataset_name,
        data_arguments.dataset_config_name,
        cache_dir=model_arguments.cache_dir,
        token=model_arguments.token,
        trust_remote_code=data_arguments.trust_remote_code,
    )
    if "validation" not in raw_datasets:
        raw_datasets["validation"] = load_dataset(
            data_arguments.dataset_name,
            data_arguments.dataset_config_name,
            split=f"train[:{data_arguments.validation_split_percentage}%]",
            cache_dir=model_arguments.cache_dir,
            token=model_arguments.token,
            trust_remote_code=data_arguments.trust_remote_code,
        )
        raw_datasets["train"] = load_dataset(
            data_arguments.dataset_name,
            data_arguments.dataset_config_name,
            split=f"train[{data_arguments.validation_split_percentage}%:]",
            cache_dir=model_arguments.cache_dir,
            token=model_arguments.token,
            trust_remote_code=data_arguments.trust_remote_code,
        )

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_arguments.cache_dir,
        "revision": model_arguments.model_revision,
        "token": model_arguments.token,
    }
    if model_arguments.config_name:
        config = AutoConfig.from_pretrained(model_arguments.config_name, **config_kwargs)
    elif model_arguments.model_name_or_path:
        config = AutoConfig.from_pretrained(model_arguments.model_name_or_path, **config_kwargs)
    else:
        config = XLNetConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_arguments.config_overrides is not None:
            logger.info(f"Overriding config: {model_arguments.config_overrides}")
            config.update_from_string(model_arguments.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_arguments.cache_dir,
        "use_fast": model_arguments.use_fast_tokenizer,
        "revision": model_arguments.model_revision,
        "token": model_arguments.token,
    }
    if model_arguments.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_arguments.tokenizer_name, **tokenizer_kwargs)
    elif model_arguments.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_arguments.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_arguments.model_name_or_path:
        model = XLNetLMHeadModel.from_pretrained(
            model_arguments.model_name_or_path,
            from_tf=bool(".ckpt" in model_arguments.model_name_or_path),
            config=config,
            cache_dir=model_arguments.cache_dir,
            revision=model_arguments.model_revision,
            token=model_arguments.token,
        )
    else:
        logger.info("Training new model from scratch")
        model = XLNetLMHeadModel(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_arguments.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_arguments.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_arguments.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_arguments.max_seq_length, tokenizer.model_max_length)

    if data_arguments.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_arguments.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(examples["text"], padding=padding, truncation=True, max_length=max_seq_length)

        with training_arguments.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_arguments.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_arguments.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with training_arguments.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_arguments.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_arguments.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.

        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_arguments.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_arguments.preprocessing_num_workers,
                load_from_cache_file=not data_arguments.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if training_arguments.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_arguments.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_arguments.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_arguments.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_arguments.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_arguments.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Data collator
    data_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer,
        plm_probability=data_arguments.plm_probability,
        max_span_length=data_arguments.max_span_length,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset if training_arguments.do_train else None,
        eval_dataset=eval_dataset if training_arguments.do_eval else None,
        processing_classtokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_arguments.do_train:
        checkpoint = None
        if training_arguments.resume_from_checkpoint is not None:
            checkpoint = training_arguments.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_arguments.max_train_samples if data_arguments.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_arguments.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_arguments.max_eval_samples if data_arguments.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_arguments.model_name_or_path, "tasks": "language-modeling"}
    if data_arguments.dataset_name is not None:
        kwargs["dataset_tags"] = data_arguments.dataset_name
        if data_arguments.dataset_config_name is not None:
            kwargs["dataset_args"] = data_arguments.dataset_config_name
            kwargs["dataset"] = f"{data_arguments.dataset_name} {data_arguments.dataset_config_name}"
        else:
            kwargs["dataset"] = data_arguments.dataset_name

    if training_arguments.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def multi_process_wrapper(index):
    """
    Wrapper for multi-processing support (for TPU/spawn).
    
    For xla_spawn (TPUs)
    """
    main()


if __name__ == "__main__":
    main()