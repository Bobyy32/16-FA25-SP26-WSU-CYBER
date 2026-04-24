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

"""Permutation LM fine-tuning training pipeline."""

from dataclasses import dataclass, field
from itertools import chain
import logging
from typing import Optional
import math
import os
import sys
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


@dataclass
class ModelArguments:
    """Settings for model initialization, config selection, and tokenizer loading."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to checkpoint for weight initialization. Skip when training from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained configuration name or path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Modify existing configuration settings when training from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if different"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Location to store downloaded pretrained models from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Enable fast tokenizer (tokenizers library) or standard one."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Model version to load (branch, tag, or commit)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "HTTP bearer token for remote files. Auto-loads from `~/.huggingface` if not specified."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "Can't combine --config_overrides with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """Data configuration for training and evaluation."""

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name from the datasets library."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Configuration name for the dataset."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Execute code from Hub repositories. Enable only for trusted sources with reviewed code."
            )
        },
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "Training data file path"})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional evaluation data file for perplexity measurement."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite cached training and evaluation datasets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "Percentage of train set used as validation when no split exists"
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "Maximum total input sequence length after tokenization. Longer sequences get truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing operations."},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": (
                "Ratio of masked span length to surrounding context length for PLM."
            )
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum token span length for masking in PLM."}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Handle distinct lines as separate sequences"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Pad all samples to `max_seq_length`. If disabled, dynamic batching padding occurs."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit training examples for debugging or faster iteration."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit evaluation examples for debugging or faster iteration."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` must be csv, json or txt."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` must be csv, json or txt."


def main():
    """Orchestrate model training for permutation language modeling."""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure logging output
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log system details
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training active: {training_args.parallel_mode.value == 'distributed'}, 16-bit training: {training_args.fp16}"
    )
    logger.info(f"Training configuration loaded: {training_args}")

    # Initialize random seed
    set_seed(training_args.seed)

    # Load datasets from Hub or local files
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=data_args.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=data_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=data_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        # Add validation split if needed
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # Load model configuration and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = XLNetConfig()
        logger.warning("Creating new configuration from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Configuration overrides applied: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"Updated configuration: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "Must provide --tokenizer_name when instantiating new tokenizer from scratch."
        )

    if model_args.model_name_or_path:
        model = XLNetLMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
        )
    else:
        logger.info("Initializing new model from configuration")
        model = XLNetLMHeadModel(config)

    # Adjust embeddings if vocab is larger than embedding layer
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Process datasets for training/evaluation
    column_names = raw_datasets["train"].column_names if training_args.do_train else raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length ({data_args.max_seq_length}) exceeds tokenizer max length. Adjusting to {tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Tokenize line-by-line or whole texts
    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Filter empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(examples["text"], padding=padding, truncation=True, max_length=max_seq_length)

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    # Prepare training and evaluation datasets
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Create data collator for PLM
    data_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer,
        plm_probability=data_args.plm_probability,
        max_span_length=data_args.max_span_length,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_classtokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Execute training phase
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Run evaluation phase
    if training_args.do_eval:
        logger.info("*** Evaluation completed ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prepare model card metadata
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "language-modeling"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # Upload to hub or create model card
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    """Multiprocessing entry point for TPU support."""
    main()


if __name__ == "__main__":
    main()