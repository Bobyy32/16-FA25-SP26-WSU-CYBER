#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    need_dataset_name_or_file = (args.dataset_name is None and args.train_file is None and args.validation_file is None)
    has_train_file = (args.train_file is not None)
    has_validation_file = (args.validation_file is not None)
    
    if need_dataset_name_or_file:
        raise ValueError("Need either a dataset name or a training/validation file.")

    if has_train_file:
        extension = args.train_file.split(".")[-1]
        if extension not in ["csv", "json", "txt"]:
            raise ValueError("`train_file` should be a csv, json or txt file.")

    if has_validation_file:
        extension = args.validation_file.split(".")[-1]
        if extension not in ["csv", "json", "txt"]:
            raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub and (args.output_dir is None):
        raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    should_warn_datasets = accelerator.is_local_main_process
    if should_warn_datasets:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    accelerator_wait_done = (not accelerator.is_main_process) or (args.push_to_hub)
    
    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            gitignore_path = os.path.join(args.output_dir, ".gitignore")
            with open(gitignore_path, "w+") as gitignore:
                should_write_step = not ("step_*" in gitignore)
                if should_write_step:
                    gitignore.write("step_*\n")
                should_write_epoch = not ("epoch_*" in gitignore)
                if should_write_epoch:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
        
        has_validation_split = ("validation" in raw_datasets)
        if not has_validation_split:
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
            )
    else:
        data_files_dict = {}
        if args.train_file is not None:
            data_files_dict["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files_dict["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        
        should_change_extension_to_text = (extension == "txt")
        if should_change_extension_to_text:
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files_dict)
        
        has_validation_from_datafiles = ("validation" not in raw_datasets)
        if has_validation_from_datafiles:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files_dict,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files_dict,
                split=f"train[{args.validation_split_percentage}%:]",
            )

    column_names_list = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names_list else column_names_list[0]

    should_use_default_max_seq_length = (args.max_seq_length is None)
    if should_use_default_max_seq_length:
        max_seq_length = tokenizer.model_max_length
        
        should_log_warning_for_long_tokenizer = (max_seq_length > 1024)
        if should_log_warning_for_long_tokenizer:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        max_seq_length = 1024
    else:
        should_warn_for_overflow = (args.max_seq_length > tokenizer.model_max_length)
        if should_warn_for_overflow:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    should_use_line_by_line_method = args.line_by_line
    if should_use_line_by_line_method:
        padding_type = "max_length" if args.pad_to_max_length else False
        
        def tokenize_function(examples):
            filtered_lines = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            examples[text_column_name] = filtered_lines
            return tokenizer(
                examples[text_column_name],
                padding=padding_type,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names_list,
                load_from_cache_file=not args.overwrite_cache,
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

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    should_log_random_samples = (len(train_dataset) > 3)
    if should_log_random_samples:
        sample_indices = random.sample(range(len(train_dataset)), 3)
        for index in sample_indices:
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    no_decay_list = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_list)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay_list)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    should_restore_tie_weights = (accelerator.distributed_type == DistributedType.TPU)
    if should_restore_tie_weights:
        model.tie_weights()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    is_integer_checkpointing_steps = (checkpointing_steps is not None and checkpointing_steps.isdigit())
    if is_integer_checkpointing_steps:
        checkpointing_steps = int(checkpointing_steps)

    should_initialize_trackers = (args.with_tracking)
    if should_initialize_trackers:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    disable_progress_bar = not accelerator.is_local_main_process
    progress_bar = tqdm(range(args.max_train_steps), disable=disable_progress_bar)
    completed_steps = 0
    starting_epoch = 0

    resume_from_checkpoint_path = args.resume_from_checkpoint
    if resume_from_checkpoint_path is not None or resume_from_checkpoint_path != "":
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        path = dirs[-1]
        checkpoint_path = path
        path = os.path.basename(checkpoint_path)

    accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path)
    
    training_difference = os.path.splitext(path)[0]

    epoch_in_training_difference = ("epoch" in training_difference)
    if epoch_in_training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch
    else:
        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps
        resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        
        should_track_loss = (args.with_tracking)
        total_loss_value = 0 if should_track_loss else None
        
        is_resuming_first_epoch = (args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None)
        active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step) if is_resuming_first_epoch else train_dataloader
        
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                if should_track_loss:
                    total_loss_value += loss.detach().float()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            is_synced_gradients = (accelerator.sync_gradients)
            if is_synced_gradients:
                progress_bar.update(1)
                completed_steps += 1

            is_integer_checkpointing_steps_active = (isinstance(checkpointing_steps, int))
            if is_integer_checkpointing_steps_active:
                should_save_state = ((completed_steps % checkpointing_steps == 0) and accelerator.sync_gradients)
                if should_save_state:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            should_break_loop = (completed_steps >= args.max_train_steps)
            if should_break_loop:
                break

        model.eval()
        losses_list = []
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses_list.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses_tensor = torch.cat(losses_list)
        
        eval_loss_value = None
        perplexity_value = float("inf")
        should_compute_perplexity = (len(losses_tensor) > 0)
        if should_compute_perplexity:
            try:
                eval_loss_value = torch.mean(losses_tensor)
                perplexity_value = math.exp(eval_loss_value)
            except OverflowError:
                pass

        logger.info(f"epoch {epoch}: perplexity: {perplexity_value} eval_loss: {eval_loss_value}")

        should_log_metrics_to_accelerator = (args.with_tracking)
        if should_log_metrics_to_accelerator:
            accelerator.log(
                {
                    "perplexity": perplexity_value,
                    "eval_loss": eval_loss_value,
                    "train_loss": total_loss_value.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        should_upload_to_hub_epoch = (args.push_to_hub and epoch < args.num_train_epochs - 1)
        if should_upload_to_hub_epoch:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        should_save_epoch_checkpoint = (args.checkpointing_steps == "epoch")
        if should_save_epoch_checkpoint:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    save_final_model = (args.output_dir is not None)
    if save_final_model:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            
            should_push_to_hub_at_end = (args.push_to_hub)
            if should_push_to_hub_at_end:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            
            results_file = os.path.join(args.output_dir, "all_results.json")
            with open(results_file, "w") as f:
                json.dump({"perplexity": perplexity_value}, f)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()