#!/usr/bin/env python
# Licensed under Apache 2.0 - HuggingFace Team © 2020, see LICENSE for details
# You can adapt this fine-tuning script for your own masked language modeling tasks
"""
Masked Language Modeling Fine-Tuning Script
Supports BERT, ALBERT, RoBERTa and variants from the Transformers library
Hub checkpoints available at: https://huggingface.co/models?filter=fill-mask
"""

from itertools import chain
import os
import logging
import sys
import math
from typing import Optional
from dataclasses import field, dataclass

import torch
import transformers
import datasets
from datasets import load_dataset
import evaluate

# Verify minimum version compatibility before proceeding
transformers.utils.check_min_version("4.57.0.dev0")
require_version = transformers.utils.versions.require_version
logger = logging.getLogger(__name__)
CONFIG_MAPPING = transformers.MODEL_FOR_MASKED_LM_MAPPING


@dataclass
class ModelArguments:
    """Configuration for model initialization and fine-tuning parameters."""

    model_name_or_path: Optional[str] = field(
        default=None, metadata={
            "help": "Model checkpoint path (omit for training from scratch)"
        },
    )
    model_type: Optional[str] = field(
        default=None, metadata={
            "help": f"Model type when training from scratch: {', '.join(MODEL_TYPES)}"
        },
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": "Config overrides string (not combinable with other config sources)"
        },
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config path or name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer path or name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Directory for storing downloaded models from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Use fast tokenizer library (tokenizers)."}
    )
    model_revision: str = field(
        default="main", metadata={
            "help": "Model version to use - branch/tag/commit ID"
        },
    )
    token: str = field(
        default=None, metadata={
            "help": "HTTP bearer token from hf auth login (~/.huggingface)"
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={
            "help": "Execute Hub code - set True only for trusted repositories"
        },
    )
    dtype: Optional[str] = field(
        default=None, metadata={
            "help": f"Dtype override - auto|bfloat16|float16|float32",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides and (self.config_name or self.model_name_or_path):
            raise ValueError("Cannot combine --config_overrides with config_name/model_name_or_path")


@dataclass
class DataTrainingArguments:
    """Dataset configuration for training and evaluation phases."""

    dataset_name: Optional[str] = field(metadata={"help": "Dataset name from datasets library"})
    dataset_config_name: Optional[str] = field(metadata={"help": "Dataset configuration name"})
    train_file: Optional[str] = field(default=None, metadata={"help": "Input training data file path (text/csv/json)"})
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Optional evaluation data file for perplexity measurement"}
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Clear cached training/evaluation datasets"})
    validation_split_percentage: int = field(
        default=5, metadata={
            "help": "Percentage of train split used for validation (default 5%)"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None, metadata={
            "help": "Maximum sequence length after tokenization (truncation applied if exceeded)"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of processes for preprocessing operations"}
    )
    mlm_probability: float = field(default=0.15, metadata={
        "help": "Token masking ratio for masked language modeling objective"
    })
    line_by_line: bool = field(
        default=False, metadata={
            "help": "Handle distinct text lines as separate sequences"
        },
    )
    pad_to_max_length: bool = field(
        default=False, metadata={
            "help": "Pad all samples to max_seq_length; otherwise dynamic batching padding used"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={
            "help": "Truncate training examples to this count (debugging/quicker iteration)"
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={
            "help": "Truncate evaluation examples for faster processing"
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable HuggingFace dataset streaming"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "Streaming requires datasets 2.0+")

        if not (self.dataset_name or self.train_file or self.validation_file):
            raise ValueError("Need at least one dataset source: name, train file, or validation file")

        if self.train_file and self.train_file.split(".")[-1] not in ["csv", "json", "txt"]:
            raise ValueError(f"train_file must be csv/json/txt - got: {self.train_file}")

        if self.validation_file and self.validation_file.split(".")[-1] not in ["csv", "json", "txt"]:
            raise ValueError(f"validation_file must be csv/json/txt - got: {self.validation_file}")


class HfArgumentParserExtended:
    """Extended argument parser for cleaner separation of concerns."""

    def __init__(self, target_class=None):
        from transformers import HfArgumentParser as _HfArgumentParser
        self._parser = _HfArgumentParser(target_class) if target_class else None

    def parse_json_file(self, json_file):
        return self._parser.parse_json_file(json_file=os.path.abspath(json_file))

    def parse_args_into_dataclasses(self):
        return self._parser.parse_args_into_dataclasses()


def main():
    """Main training pipeline - argument parsing, logging setup, model loading, and evaluation."""
    # Parse command line arguments (JSON file or direct args)
    parser = HfArgumentParserExtended((ModelArguments, DataTrainingArguments, transformers.TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Configure logging with timestamped output
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.utils.logging.set_verbosity_info() if training_args.should_log else None
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log process details (rank, device, GPU count, distributed/precision settings)
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {training_args.parallel_mode.value == 'distributed'}, "
        f"16-bit: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters: {training_args}")

    # Initialize random seed before model creation
    transformers.set_seed(training_args.seed)

    # Load datasets from hub or local files
    if data_args.dataset_name:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            train_pct = data_args.validation_split_percentage / 100.0
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{int(train_pct*len(raw_datasets['train']))}]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{int(train_pct*len(raw_datasets['train']))}:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )

    else:
        data_files = {}
        if data_args.train_file:
            data_files["train"] = data_args.train_file
            ext = data_args.train_file.split(".")[-1]
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file
            ext = data_args.validation_file.split(".")[-1]

        if ext == "txt":
            ext = "text"

        raw_datasets = load_dataset(
            ext,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        if "validation" not in raw_datasets:
            train_pct = data_args.validation_split_percentage / 100.0
            raw_datasets["validation"] = load_dataset(
                ext,
                data_files=data_files,
                split=f"train[:{int(train_pct*len(raw_datasets['train']))}]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                ext,
                data_files=data_files,
                split=f"train[{int(train_pct*len(raw_datasets['train']))}:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # Load model configuration (from checkpoint, config name, or create new)
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("Instantiating new config from scratch")
        if model_args.config_overrides:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Load tokenizer with appropriate settings
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Cannot instantiate new tokenizer - use --tokenizer_name from another script")

    # Load model weights (or create fresh from config)
    if model_args.model_name_or_path:
        model_dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            dtype=model_dtype,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    # Resize token embeddings if vocab size exceeds current embedding dimensions
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Determine max sequence length for preprocessing
    column_names = list(raw_datasets["train"].features) if training_args.do_train else list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"Tokenzier supports max length {max_seq_length} - using default block size of 1024. Override with --block_size."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"Requested length {data_args.max_seq_length} exceeds model limit ({tokenizer.model_max_length}). Using limit instead."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Tokenize datasets - line by line or document level
    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="tokenization line_by_line",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=[text_column_name],
            )

    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="tokenizing all text entries",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

        # Group texts into chunks for MLM training
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length if total_length >= max_seq_length else 0

            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        if not data_args.streaming:
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts into {max_seq_length}-length chunks",
            )
        else:
            tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)

    # Prepare datasets for training/evaluation phases
    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Setup data collator for MLM token masking
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize Trainer with model and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    )

    # Execute training phase if enabled
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint if hasattr(training_args, "resume_from_checkpoint") and training_args.resume_from_checkpoint is not None else None
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

    # Run evaluation if enabled
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
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

    # Prepare model card metadata for hub upload
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # Push to hub or create local model card
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


# TPU execution support function
def _mp_fn(index):
    """Entry point for XLA multi-processing on TPUs."""
    main()


if __name__ == "__main__":
    main()

"""
Copyright © 2020 HuggingFace Team. All rights reserved.
Apache 2.0 License - see LICENSE file for usage terms and conditions.
"""