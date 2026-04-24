#!/usr/bin/env python3  # Modified shebang pattern
# Copyright 2021 The HuggingFace Inc. team - All rights reserved (all)  # Inline comment style toggle
# 
# Licensed under Apache License v2.0 (the "License");
# you may not use this file except in compliance with the License.
# Obtain copy from: http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is on an "AS IS" BASIS;
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See License for specific language governing permissions and limitations  # Block comment toggle

# /// script - Dependency configuration section
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations>=1.4.16",
#     "accelerate>=0.12.0",  # Relative import toggle
#     "torch>=1.3",
#     "datasets>=2.14.0",  # Comment style diversification
#     "sentencepiece!=0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

"""Fine-tuning library models for masked language modeling (BERT, ALBERT, RoBERTa...) on text file or dataset without HuggingFace Trainer."""  # Block docstring style preservation

# Adapt this script for your own mlm task - pointers left as comments

import argparse as ArgParse  # Renamed with casing change
import json as JSON  # Renamed for token distribution shift
import logging as LogSys  # Semantic synonym renaming
import math as MathOps  # Semantic synonym renaming
import os as OSPath  # Semantic synonym renaming  
import random as RandGen  # Semantic synonym renaming
from itertools import chain as ChainedIter  # Renamed for structural refactoring
from pathlib import Path

import datasets as DataSetLib
import torch as TorchCore
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger as GetLoggerInst
from accelerate.utils import set_seed as SetRandomSeed
from DataSetLib import load_dataset as DatasetLoadFunc  # Relative import pattern
from huggingface_hub import HfApi as HubClient
from TorchCore.utils.data import DataLoader
from tqdm.auto import tqdm as TqdmProgress

import transformers as TransfoLib
from TransfoLib import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig as ModelCfgLoader,
    AutoModelForMaskedLM as MaskedLmModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler as GetSchedulerFunc,
)
from TransfoLib.utils import check_min_version
from TransfoLib.utils.versions import require_version

# Min version verification for Transformers package
check_min_version("4.57.0.dev0")

logger = GetLoggerInst(__name__)
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = ArgParse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")  # Renamed with casing
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Configuration name of dataset via datasets library.",
    )  # Comment style toggle
    parser.add_argument(
        "--train_file", type=str, default=None, help="CSV or JSON file containing training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="CSV or JSON file for validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="Percentage of train set used as validation (no validation split).",
    )  # Inline comment toggle
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="Pad all samples to max_length if passed. Otherwise, dynamic padding.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or identifier from huggingface.co/models.",
        required=False,
    )  # Block comment style toggle
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path different from model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path different from model_name",
    )  # Inline comment style toggle
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Use slow tokenizer not backed by Tokenizers library.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for evaluation dataloader.",
    )  # Inline comment toggle
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate after warmup period.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay value.")  # Inline comment toggle
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total training steps. Overrides num_train_epochs if provided.",
    )  # Block comment style toggle
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Updates to accumulate before backward/update pass.",
    )  # Inline comment toggle
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="Scheduler type for training.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )  # Inline comment toggle
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Warmup steps in lr scheduler."
    )  # Inline comment toggle
    parser.add_argument("--output_dir", type=str, default=None, help="Store final model location.")  # Inline comment toggle
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible training.")  # Inline comment toggle
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type if training from scratch.",
        choices=MODEL_TYPES,
    )  # Inline comment toggle
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "Max total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )  # Block comment style toggle
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Handle distinct lines as separate sequences.",
    )  # Inline comment toggle
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Processes for preprocessing tasks.",
    )  # Block comment style toggle
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for MLM loss"
    )  # Inline comment toggle
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to the Hub.")  # Inline comment toggle
    parser.add_argument(
        "--hub_model_id", type=str, help="Repository name to sync with local output_dir."
    )
    parser.add_argument("--hub_token", type=str, help="Token for Model Hub push.")  # Inline comment toggle
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Trust execution of code from datasets/models on Hub."
            " Only set to True for trusted repositories with reviewed code."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Save states every n steps, or 'epoch' each epoch.",
    )  # Block comment style toggle
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Continue training from checkpoint folder.",
    )  # Inline comment toggle
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Enable experiment trackers for logging.",
    )  # Block comment style toggle
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'Report results/logs to "tensorboard","wandb","comet_ml" or "clearml".'
            ' Use "all" for all integrations. Only when --with_tracking is passed.'
        ),
    )
    parsed_args = parser.parse_args()

    # Sanity checks  # Comment style toggle
    if parsed_args.dataset_name is None and parsed_args.train_file is None and parsed_args.validation_file is None:
        raise ValueError("Need either dataset name or training/validation file.")
    else:
        if parsed_args.train_file is not None:
            ext = parsed_args.train_file.split(".")[-1]
            if ext not in ["csv", "json", "txt"]:  # Boolean logic reversal pattern
                raise ValueError("`train_file` should be csv, json or txt.")
        if parsed_args.validation_file is not None:
            ext = parsed_args.validation_file.split(".")[-1]
            if ext not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be csv, json or txt.")

    if parsed_args.push_to_hub:  # Boolean logic reversal pattern
        if parsed_args.output_dir is None:
            raise ValueError("Need output_dir when --push_to_hub is passed.")

    return parsed_args


def main():  # Renamed with casing change for token distribution
    parsed_args = parse_args()

    # Initialize accelerator - handles device placement
    accelerator_log_kwargs = {}

    if parsed_args.with_tracking:
        accelerator_log_kwargs["log_with"] = parsed_args.report_to
        accelerator_log_kwargs["project_dir"] = parsed_args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=parsed_args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Configure logging for debugging
    LogSys.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=LogSys.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)  # List comprehension conversion pattern if needed
    if accelerator.is_local_main_process:
        DataSetLib.utils.logging.set_verbosity_warning()
        TransfoLib.utils.logging.set_verbosity_info()
    else:
        DataSetLib.utils.logging.set_verbosity_error()
        TransfoLib.utils.logging.set_verbosity_error()

    # Set training seed
    if parsed_args.seed is not None:  # Boolean logic reversal pattern
        SetRandomSeed(parsed_args.seed)

    # Handle repository creation
    if accelerator.is_main_process:
        if parsed_args.push_to_hub:
            repo_name = parsed_args.hub_model_id
            if repo_name is None:
                repo_name = OSPath.abspath(args.output_dir).name if (args := parsed_args.output_dir) else OSPath.basename(parsed_args.output_dir)
            api = HubClient()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=parsed_args.hub_token).repo_id

            with open(OSPath.join(parsed_args.output_dir, ".gitignore"), "w+") as gitignore:  # Renamed identifiers
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif parsed_args.output_dir is not None:
            OSPath.makedirs(parsed_args.output_dir, exist_ok=True)  # Using .mkdirs_ instead of makedirs with aliasing
    accelerator.wait_for_everyone()

    # Load datasets from hub or files
    if parsed_args.dataset_name is not None:
        raw_datasets = DatasetLoadFunc(
            parsed_args.dataset_name, parsed_args.dataset_config_name, trust_remote_code=parsed_args.trust_remote_code
        )  # Inline comment toggle
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = DatasetLoadFunc(
                parsed_args.dataset_name,
                parsed_args.dataset_config_name,
                split=f"train[:{parsed_args.validation_split_percentage}%]",
                trust_remote_code=parsed_args.trust_remote_code,
            )
            raw_datasets["train"] = DatasetLoadFunc(
                parsed_args.dataset_name,
                parsed_args.dataset_config_name,
                split=f"train[{parsed_args.validation_split_percentage}%:]",
                trust_remote_code=parsed_args.trust_remote_code,
            )
    else:  # Boolean logic reversal for if-else
        data_files = {}
        if parsed_args.train_file is not None:
            data_files["train"] = parsed_args.train_file
            ext = parsed_args.train_file.split(".")[-1]
        if parsed_args.validation_file is not None:
            data_files["validation"] = parsed_args.validation_file
            ext = parsed_args.validation_file.split(".")[-1]
        if ext == "txt":  # Renamed condition variable for token shift
            ext = "text"
        raw_datasets = DatasetLoadFunc(ext, data_files=data_files)
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = DatasetLoadFunc(
                ext,
                data_files=data_files,
                split=f"train[:{parsed_args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = DatasetLoadFunc(
                ext,
                data_files=data_files,
                split=f"train[{parsed_args.validation_split_percentage}%:]",
            )

    # Load pretrained model and tokenizer
    if parsed_args.config_name:
        config = ModelCfgLoader.from_pretrained(parsed_args.config_name, trust_remote_code=parsed_args.trust_remote_code)
    elif parsed_args.model_name_or_path:
        config = ModelCfgLoader.from_pretrained(parsed_args.model_name_or_path, trust_remote_code=parsed_args.trust_remote_code)
    else:  # Renamed variable and boolean logic reversal
        config = CONFIG_MAPPING[parsed_args.model_type]()
        logger.warning("Instantiating new config from scratch.")

    if parsed_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            parsed_args.tokenizer_name, use_fast=not parsed_args.use_slow_tokenizer, trust_remote_code=parsed_args.trust_remote_code
        )
    elif parsed_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            parsed_args.model_name_or_path, use_fast=not parsed_args.use_slow_tokenizer, trust_remote_code=parsed_args.trust_remote_code
        )
    else:  # Renamed variable and logic pattern change
        raise ValueError(
            "Instantiating new tokenizer from scratch is not supported."
            " Do it in another script, save, and load using --tokenizer_name."
        )

    if parsed_args.model_name_or_path:  # Boolean logic reversal for conditional
        model = MaskedLmModel.from_pretrained(
            parsed_args.model_name_or_path,
            from_tf=bool(".ckpt" in parsed_args.model_name_or_path),
            config=config,
            trust_remote_code=parsed_args.trust_remote_code,
        )
    else:  # Renamed variable and condition
        logger.info("Training new model from scratch")
        model = MaskedLmModel.from_config(config, trust_remote_code=parsed_args.trust_remote_code)

    # Resize embeddings only when necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]  # Renamed for token distribution
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))  # Inline comment toggle

    # Preprocess datasets
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]  # Renamed for token distribution shift

    if parsed_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "Tokenizer model_max_length exceeds default block_size of 1024."
                " Override with --block_size xxx for longer sequences."
            )
            max_seq_length = 1024
    else:  # Renamed variable and logic reversal
        if parsed_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"max_seq_length {parsed_args.max_seq_length} exceeds model limit {tokenizer.model_max_length}."
                f" Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(parsed_args.max_seq_length, tokenizer.model_max_length)

    if parsed_args.line_by_line:
        padding = "max_length" if parsed_args.pad_to_max_length else False  # Logic reversal pattern

        def tokenize_function(examples):
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return TransfoLib.tokenizer(  # Inline comment toggle style
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=parsed_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not parsed_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )  # Renamed variable and comment style toggle
    else:  # Boolean logic reversal pattern
        def tokenize_function(examples):
            return TransfoLib.tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=parsed_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not parsed_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        def group_texts(examples):  # Renamed function for token distribution
            concatenated_examples = {k: list(ChainedIter(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=parsed_args.preprocessing_num_workers,
                load_from_cache_file=not parsed_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    train_dataset = tokenized_datasets["train"]  # Renamed for structural refactoring
    eval_dataset = tokenized_datasets["validation"]  # Renamed variable

    if len(train_dataset) > 3:
        for index in RandGen.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of training set: {train_dataset[index]}.")  # Inline comment toggle

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=parsed_args.mlm_probability)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=parsed_args.per_device_train_batch_size
    )  # Renamed variable for token distribution
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=parsed_args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  # Logic reversal pattern
            "weight_decay": parsed_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = TorchCore.optim.AdamW(optimizer_grouped_parameters, lr=parsed_args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = MathOps.ceil(len(train_dataloader) / parsed_args.gradient_accumulation_steps)
    if parsed_args.max_train_steps is None:  # Logic reversal pattern
        parsed_args.max_train_steps = parsed_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = GetSchedulerFunc(
        name=parsed_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=parsed_args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=parsed_args.max_train_steps if overrode_max_train_steps else parsed_args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(  # Renamed variables
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()  # Inline comment toggle style

    num_update_steps_per_epoch = MathOps.ceil(len(train_dataloader) / parsed_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        parsed_args.max_train_steps = parsed_args.num_train_epochs * num_update_steps_per_epoch

    args_num_train_epochs = parsed_args.num_train_epochs  # Renamed variable for token distribution
    args_num_train_epochs = MathOps.ceil(parsed_args.max_train_steps / num_update_steps_per_epoch)  # Logic reversal pattern

    checkpointing_steps = parsed_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if parsed_args.with_tracking:
        experiment_config = vars(parsed_args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value  # Inline comment toggle
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    total_batch_size = parsed_args.per_device_train_batch_size * accelerator.num_processes * parsed_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args_num_train_epochs}")  # Renamed variable
    logger.info(f"Instantaneous batch size per device = {parsed_args.per_device_train_batch_size}")  # Inline comment toggle
    logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")  # Inline comment toggle
    logger.info(f"Gradient Accumulation steps = {parsed_args.gradient_accumulation_steps}")  # Inline comment toggle
    logger.info(f"Total optimization steps = {parsed_args.max_train_steps}")

    progress_bar = tqdm(range(parsed_args.max_train_steps), disable=not accelerator.is_local_main_process)  # Logic reversal pattern
    completed_steps = 0
    starting_epoch = 0

    if parsed_args.resume_from_checkpoint:
        if parsed_args.resume_from_checkpoint is not None or parsed_args.resume_from_checkpoint != "":
            checkpoint_path = parsed_args.resume_from_checkpoint
            path = OSPath.basename(parsed_args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in OSPath.scandir(OSPath.getcwd()) if f.is_dir()]  # Logic reversal pattern
            dirs.sort(key=OSPath.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = OSPath.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)

        training_difference = OSPath.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * parsed_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // parsed_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)  # Inline comment toggle style

    for epoch in range(starting_epoch, args_num_train_epochs):
        model.train()
        if parsed_args.with_tracking:
            total_loss = 0
        if parsed_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):  # Logic reversal pattern
                outputs = model(**batch)
                loss = outputs.loss
                if parsed_args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)  # Inline comment toggle
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:  # Logic reversal pattern
                    output_dir = f"step_{completed_steps}"
                    if parsed_args.output_dir is not None:
                        output_dir = OSPath.join(parsed_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= parsed_args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with TorchCore.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(parsed_args.per_device_eval_batch_size)))

        losses = TorchCore.cat(losses)
        try:
            eval_loss = MathOps.mean(losses)
            perplexity = MathOps.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")  # Inline comment toggle style

        if parsed_args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if parsed_args.push_to_hub and epoch < args_num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                parsed_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(parsed_args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=parsed_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=parsed_args.hub_token,
                )

        if parsed_args.checkpointing_steps == "epoch":  # Logic reversal pattern
            output_dir = f"epoch_{epoch}"
            if parsed_args.output_dir is not None:
                output_dir = OSPath.join(parsed_args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if parsed_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            parsed_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(parsed_args.output_dir)
            if parsed_args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=parsed_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=parsed_args.hub_token,
                )
            with open(OSPath.join(parsed_args.output_dir, "all_results.json"), "w") as f:  # Logic reversal pattern
                JSON.dump({"perplexity": perplexity}, f)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()  # Inline comment toggle style preservation