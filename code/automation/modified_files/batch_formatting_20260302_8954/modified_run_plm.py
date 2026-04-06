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

"""
Permutation language modeling through fine-tuning operations on library models.
"""
# You can also adapt this script for your personal permutation language modeling objectives. Guidance markers are left as annotations.

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


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Configuration parameters for model, configuration, and tokenizer initialization or fresh training.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Model checkpoint path for weight initialization. Do not set when training from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained configuration identifier or location differing from model name"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Replace certain configuration defaults when initiating training from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer identifier or location if distinct from model name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory for storing pretrained models retrieved from huggingface.co"
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": (
            "Switch between fast tokenizer implementation supported by tokenizers library or standard tokenizer."
        )},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": (
            "Specific model version identifier (branch, tag, or commit identifier to use)."
        )},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "Authentication token for remote file access. If unspecified, uses the token created during `hf auth login` "
                "(stored in `~/.huggingface`). 
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides cannot be combined with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Parameters defining training/evaluation data specifications for model input.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": (
            "Dataset identifier from the datasets library to utilize."
        )}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": (
            "Configuration name of the dataset from the datasets library to use."
        )}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow execution of code from datasets/models defined on Hub. Set to `True` only for trusted repositories"
                " where code has been reviewed, as it executes remote code locally."
            )
        },
    )
    train_file: Optional[str] = field(default=None, metadata={"help": (
        "Input training data file path (text file format)."
    )})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Optional evaluation data file path for perplexity measurement (text file format)."
        )},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Clear cached training and evaluation datasets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Proportion of train data allocated to validation set when no validation split exists."
            )
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "Maximum sequence length after tokenization. Sequences exceeding this limit get truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": (
            "Worker process count for preprocessing operations."
        )},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": (
                "Fraction of masked token span length relative to surrounding context in permutation language modeling."
            )
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": (
        "Upper bound for consecutive masked tokens in permutation language modeling spans."
    )}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": (
            "Treat individual text lines as separate sequences when enabled."
        )},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Pad all samples to `max_seq_length`. When disabled, dynamic batching adjustment occurs at maximum batch length."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging or expedited training, cap training examples to this value when specified."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging or expedited evaluation, limit evaluation examples to this value when set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Provide either dataset name or training/evaluation file paths.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` must be csv, json, or txt format."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` must be csv, json, or txt format."


def main():
    # Reference available arguments in src/transformers/training_args.py
    # or by passing --help to this script.
    # We maintain separate argument sets for clearer concern separation.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # When single argument passed as JSON file path, parse it to extract arguments.
        model_args_obj, dataset_training_args, trainer_arguments = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args_obj, dataset_training_args, trainer_arguments = parser.parse_args_into_dataclasses()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if trainer_arguments.should_log:
        # Default training_args.log_level is passive, so set verbosity to info here.
        transformers.utils.logging.set_verbosity_info()

    log_level = trainer_arguments.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log process summary on each process:
    logger.warning(
        f"Process rank: {trainer_arguments.local_process_index}, device: {trainer_arguments.device}, n_gpu: {trainer_arguments.n_gpu}, "
        + f"distributed training: {trainer_arguments.parallel_mode.value == 'distributed'}, 16-bits training: {trainer_arguments.fp16}"
    )
    logger.info(f"Training/evaluation parameters {trainer_arguments}")

    # Set seed before model initialization.
    set_seed(trainer_arguments.seed)

    # Retrieve datasets: provide custom CSV/JSON/TXT files or public dataset from hub.
    # Public datasets are automatically downloaded from datasets Hub.
    #
    # For CSV/JSON files, use 'text' column or first available column if no 'text' column exists.
    #
    # In distributed training, load_dataset ensures only one local process downloads the dataset concurrently.
    if dataset_training_args.dataset_name is not None:
        # Load dataset from hub.
        raw_datasets_collection = load_dataset(
            dataset_training_args.dataset_name,
            dataset_training_args.dataset_config_name,
            cache_dir=model_args_obj.cache_dir,
            token=model_args_obj.token,
            trust_remote_code=dataset_training_args.trust_remote_code,
        )
        if "validation" not in raw_datasets_collection:
            raw_datasets_collection["validation"] = load_dataset(
                dataset_training_args.dataset_name,
                dataset_training_args.dataset_config_name,
                split=f"train[:{dataset_training_args.validation_split_percentage}%]",
                cache_dir=model_args_obj.cache_dir,
                token=model_args_obj.token,
                trust_remote_code=dataset_training_args.trust_remote_code,
            )
            raw_datasets_collection["train"] = load_dataset(
                dataset_training_args.dataset_name,
                dataset_training_args.dataset_config_name,
                split=f"train[{dataset_training_args.validation_split_percentage}%:]",
                cache_dir=model_args_obj.cache_dir,
                token=model_args_obj.token,
                trust_remote_code=dataset_training_args.trust_remote_code,
            )
    else:
        data_files = {}
        if dataset_training_args.train_file is not None:
            data_files["train"] = dataset_training_args.train_file
            extension = dataset_training_args.train_file.split(".")[-1]
        if dataset_training_args.validation_file is not None:
            data_files["validation"] = dataset_training_args.validation_file
            extension = dataset_training_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets_collection = load_dataset(extension, data_files=data_files, cache_dir=model_args_obj.cache_dir)
        # When no validation data exists, use validation_split_percentage to partition the dataset.
        if "validation" not in raw_datasets_collection:
            raw_datasets_collection["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{dataset_training_args.validation_split_percentage}%]",
                cache_dir=model_args_obj.cache_dir,
                token=model_args_obj.token,
            )
            raw_datasets_collection["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{dataset_training_args.validation_split_percentage}%:]",
                cache_dir=model_args_obj.cache_dir,
                token=model_args_obj.token,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_parameters = {
        "cache_dir": model_args_obj.cache_dir,
        "revision": model_args_obj.model_revision,
        "token": model_args_obj.token,
    }
    if model_args_obj.config_name:
        model_configuration = AutoConfig.from_pretrained(model_args_obj.config_name, **config_parameters)
    elif model_args_obj.model_name_or_path:
        model_configuration = AutoConfig.from_pretrained(model_args_obj.model_name_or_path, **config_parameters)
    else:
        model_configuration = XLNetConfig()
        logger.warning("Instantiating new config instance from scratch.")
        if model_args_obj.config_overrides is not None:
            logger.info(f"Overriding configuration: {model_args_obj.config_overrides}")
            model_configuration.update_from_string(model_args_obj.config_overrides)
            logger.info(f"New configuration: {model_configuration}")

    tokenization_parameters = {
        "cache_dir": model_args_obj.cache_dir,
        "use_fast": model_args_obj.use_fast_tokenizer,
        "revision": model_args_obj.model_revision,
        "token": model_args_obj.token,
    }
    if model_args_obj.tokenizer_name:
        tokenization_object = AutoTokenizer.from_pretrained(model_args_obj.tokenizer_name, **tokenization_parameters)
    elif model_args_obj.model_name_or_path:
        tokenization_object = AutoTokenizer.from_pretrained(model_args_obj.model_name_or_path, **tokenization_parameters)
    else:
        raise ValueError(
            "Instantiating new tokenizer from scratch. This is not supported by this script. "
            "Perform it separately, save it, and load using --tokenizer_name."
        )

    if model_args_obj.model_name_or_path:
        XLNetLMHeadModel.from_pretrained(
            model_args_obj.model_name_or_path,
            from_tf=bool(".ckpt" in model_args_obj.model_name_or_path),
            config=model_configuration,
            cache_dir=model_args_obj.cache_dir,
            revision=model_args_obj.model_revision,
            token=model_args_obj.token,
        )
    else:
        logger.info("Training new model from scratch")
        XLNetLMHeadModel(model_configuration)

    # Resize embeddings only when necessary to avoid index errors. When creating a model from scratch
    # on small vocabulary with smaller embedding size desired, remove this check.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenization_object) > embedding_size:
        model.resize_token_embeddings(len(tokenization_object))

    # Preprocessing the datasets.
    # First tokenize all texts.
    if trainer_arguments.do_train:
        column_names = raw_datasets_collection["train"].column_names
    else:
        column_names = raw_datasets_collection["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if dataset_training_args.max_seq_length > tokenization_object.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({dataset_training_args.max_seq_length}) exceeds maximum model length "
            f"({tokenization_object.model_max_length}). Using max_seq_length={tokenization_object.model_max_length}."
        )
    max_seq_length = min(dataset_training_args.max_seq_length, tokenization_object.model_max_length)

    if dataset_training_args.line_by_line:
        # With line_by_line, tokenize each nonempty line individually.
        padding = "max_length" if dataset_training_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Filter empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenization_object(examples["text"], padding=padding, truncation=True, max_length=max_seq_length)

        with trainer_arguments.main_process_first(desc="dataset map tokenization"):
            processed_dataset_collection = raw_datasets_collection.map(
                tokenize_function,
                batched=True,
                num_proc=dataset_training_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not dataset_training_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise tokenize each text then concatenate before splitting into smaller parts.
        def tokenize_function(examples):
            return tokenization_object(examples[text_column_name])

        with trainer_arguments.main_process_first(desc="dataset map tokenization"):
            processed_dataset_collection = raw_datasets_collection.map(
                tokenize_function,
                batched=True,
                num_proc=dataset_training_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not dataset_training_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main processing function concatenates all dataset texts into chunks of max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # Drop small remainder; if total_length < max_seq_length, exclude batch and return empty dict.
            # Could add padding if model supported it instead of dropping, customize as needed.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # With `batched=True`, this map processes 1,000 texts together, so group_texts discards a
        # remainder for each group of 1,000 texts. Can adjust batch_size here but higher values might slow preprocessing.
        #
        # To speed up this part, use multiprocessing. See documentation for map method:
        # https://huggingface.co/docs/datasets/process#map

        with trainer_arguments.main_process_first(desc="grouping texts together"):
            processed_dataset_collection = processed_dataset_collection.map(
                group_texts,
                batched=True,
                num_proc=dataset_training_args.preprocessing_num_workers,
                load_from_cache_file=not dataset_training_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if trainer_arguments.do_train:
        if "train" not in processed_dataset_collection:
            raise ValueError("--do_train requires train dataset")
        training_set_data = processed_dataset_collection["train"]
        if dataset_training_args.max_train_samples is not None:
            max_train_samples = min(len(training_set_data), dataset_training_args.max_train_samples)
            training_set_data = training_set_data.select(range(max_train_samples))

    if trainer_arguments.do_eval:
        if "validation" not in processed_dataset_collection:
            raise ValueError("--do_eval requires validation dataset")
        evaluation_set_data = processed_dataset_collection["validation"]
        if dataset_training_args.max_eval_samples is not None:
            max_eval_samples = min(len(evaluation_set_data), dataset_training_args.max_eval_samples)
            evaluation_set_data = evaluation_set_data.select(range(max_eval_samples))

    # Data collator
    data_collator = DataCollatorForPermutationLanguageModeling(
        tokenization_object=tokenization_object,
        plm_probability=dataset_training_args.plm_probability,
        max_span_length=dataset_training_args.max_span_length,
    )

    # Initialize trainer
    model_trainer_instance = Trainer(
        model=model,
        args=trainer_arguments,
        train_dataset=training_set_data if trainer_arguments.do_train else None,
        eval_dataset=evaluation_set_data if trainer_arguments.do_eval else None,
        processing_classtokenizer=tokenization_object,
        data_collator=data_collator,
    )

    # Training
    if trainer_arguments.do_train:
        checkpoint = None
        if trainer_arguments.resume_from_checkpoint is not None:
            checkpoint = trainer_arguments.resume_from_checkpoint
        train_result = model_trainer_instance.train(resume_from_checkpoint=checkpoint)
        model_trainer_instance.save_model()  # Also saves tokenizer for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            dataset_training_args.max_train_samples if dataset_training_args.max_train_samples is not None else len(training_set_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(training_set_data))

        model_trainer_instance.log_metrics("train", metrics)
        model_trainer_instance.save_metrics("train", metrics)
        model_trainer_instance.save_state()

    # Evaluation
    if trainer_arguments.do_eval:
        logger.info("*** Evaluate ***")

        metrics = model_trainer_instance.evaluate()

        max_eval_samples = dataset_training_args.max_eval_samples if dataset_training_args.max_eval_samples is not None else len(evaluation_set_data)
        metrics["eval_samples"] = min(max_eval_samples, len(evaluation_set_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        model_trainer_instance.log_metrics("eval", metrics)
        model_trainer_instance.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args_obj.model_name_or_path, "tasks": "language-modeling"}
    if dataset_training_args.dataset_name is not None:
        kwargs["dataset_tags"] = dataset_training_args.dataset_name
        if dataset_training_args.dataset_config_name is not None:
            kwargs["dataset_args"] = dataset_training_args.dataset_config_name
            kwargs["dataset"] = f"{dataset_training_args.dataset_name} {dataset_training_args.dataset_config_name}"
        else:
            kwargs["dataset"] = dataset_training_args.dataset_name

    if trainer_arguments.push_to_hub:
        model_trainer_instance.push_to_hub(**kwargs)
    else:
        model_trainer_instance.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()