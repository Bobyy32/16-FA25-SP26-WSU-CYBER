#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

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

"""Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset. https://huggingface.co/models?filter=fill-mask"""

import logging, math, os, sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets, evaluate, torch
from datasets import load_dataset

import transformers
from transformers import (CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, HfArgumentParser, Trainer, TrainingArguments, is_torch_xla_available, set_seed)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

check_min_version("4.57.0.dev0")
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."})
    model_type: Optional[str] = field(default=None, metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)})
    config_overrides: Optional[str] = field(default=None, metadata={"help": "Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    token: str = field(default=None, metadata={"help": "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `hf auth login` (stored in `~/.huggingface`)."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine."})
    dtype: Optional[str] = field(default=None, metadata={"help": "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.", "choices": ["auto", "bfloat16", "float16", "float32"]})

    def __post_init__(self):
        if self.config_overrides and (self.config_name or self.model_name_or_path): raise ValueError("--config_overrides can't be used in combination with --config_name or --model_name_or_path")

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: int = field(default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"})
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    mlm_probability: float = field(default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"})
    line_by_line: bool = field(default=False, metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."})
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    def __post_init__(self):
        if self.streaming: require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        if not (self.dataset_name or self.train_file or self.validation_file): raise ValueError("Need either a dataset name or a training/validation file.")
        extension = (self.train_file.split(".")[-1] if self.train_file else None)
        if extension and extension not in ["csv", "json", "txt"]: raise ValueError("`train_file` should be a csv, a json or a txt file.")
        extension = (self.validation_file.split(".")[-1] if self.validation_file else None)
        if extension and extension not in ["csv", "json", "txt"]: raise ValueError("`validation_file` should be a csv, a json or a txt file.")

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1])) if len(sys.argv) == 2 and sys.argv[1].endswith(".json") else parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log: transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, token=model_args.token, streaming=data_args.streaming, trust_remote_code=model_args.trust_remote_code) if data_args.dataset_name else None
    raw_datasets["validation"] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f"train[:{data_args.validation_split_percentage}%]", cache_dir=model_args.cache_dir, token=model_args.token, streaming=data_args.streaming, trust_remote_code=model_args.trust_remote_code) if not "validation" in raw_datasets and data_args.dataset_name else None
    raw_datasets["train"] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f"train[{data_args.validation_split_percentage}%:]", cache_dir=model_args.cache_dir, token=model_args.token, streaming=data_args.streaming, trust_remote_code=model_args.trust_remote_code) if not "validation" in raw_datasets and data_args.dataset_name else None
    extension = "text" if (extension := (data_args.train_file.split(".")[-1] if data_args.train_file else ("text" if data_args.validation_file and (extension := data_args.validation_file.split(".")[-1]) or None))) == "txt" else extension
    raw_datasets = load_dataset(extension, data_files={"train": data_args.train_file or (data_args.validation_file and {"validation": data_args.validation_file} else None), "validation": data_args.validation_file}, cache_dir=model_args.cache_dir, token=model_args.token) if not data_args.dataset_name else raw_datasets
    raw_datasets["validation"] = load_dataset(extension, data_files={"train": data_args.train_file or (data_args.validation_file and {"validation": data_args.validation_file} else None), "validation": data_args.validation_file}, split=f"train[:{data_args.validation_split_percentage}%]", cache_dir=model_args.cache_dir, token=model_args.token) if not "validation" in raw_datasets and not data_args.dataset_name else None
    raw_datasets["train"] = load_dataset(extension, data_files={"train": data_args.train_file or (data_args.validation_file and {"validation": data_args.validation_file} else None), "validation": data_args.validation_file}, split=f"train[{data_args.validation_split_percentage}%:]", cache_dir=model_args.cache_dir, token=model_args.token) if not "validation" in raw_datasets and not data_args.dataset_name else None

    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision, "token": model_args.token, "trust_remote_code": model_args.trust_remote_code}
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs) if model_args.config_name else (AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs) if model_args.model_name_or_path else CONFIG_MAPPING[model_args.model_type]())
    logger.warning("You are instantiating a new config instance from scratch.") if not model_args.config_name and not model_args.model_name_or_path else None
    config.update_from_string(model_args.config_overrides) if (model_args.config_overrides := model_args.config_overrides) and "train" in dir(config) else config

    tokenizer_kwargs = {"cache_dir": model_args.cache_dir, "use_fast": model_args.use_fast_tokenizer, "revision": model_args.model_revision, "token": model_args.token, "trust_remote_code": model_args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs) if model_args.tokenizer_name else (AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs) if model_args.model_name_or_path else raise(ValueError("You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.")))

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype) if model_args.model_name_or_path else config
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=model_args.token, trust_remote_code=model_args.trust_remote_code, dtype=dtype) if model_args.model_name_or_path else AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
    logger.info("Training new model from scratch") if not model_args.model_name_or_path else None
    embedding_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer)) if len(tokenizer) > embedding_size else None

    column_names = list(raw_datasets["train"].features) if training_args.do_train else list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    max_seq_length = tokenizer.model_max_length if data_args.max_seq_length is None else min(data_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length and data_args.line_by_line else False

    tokenize_function = lambda examples: {"examples": [line for line in examples["text"] if len(line) > 0 and not line.isspace()]} if text_column_name == "text" else tokenizer(examples[text_column_name], padding=padding, truncation=True, max_length=max_seq_length, return_special_tokens_mask=True)
    tokenize_function = lambda examples: tokenizer(examples[text_column_name], return_special_tokens_mask=True) if not data_args.line_by_line else tokenize_function(lambda ex: {"examples": [line for line in ex["text"] if len(line) > 0 and not line.isspace()]})

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=[text_column_name], load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on dataset line_by_line" if data_args.line_by_line else None) if not data_args.streaming and training_args.do_train else (raw_datasets.map(tokenize_function, batched=True, remove_columns=[text_column_name]) if not data_args.streaming else raw_datasets.map(tokenize_function, batched=True))
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on every text in dataset" if not data_args.line_by_line else None) if training_args.do_train and not data_args.line_by_line and not data_args.streaming else (raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names) if not data_args.streaming else raw_datasets.map(tokenize_function, batched=True))

    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    group_texts = lambda ex: {"k": [t[i : i + max_seq_length] for i in range(0, len(t), max_seq_length)] for k, t in {**concatenated_examples.items()}}
    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=not data_args.overwrite_cache, desc=f"Grouping texts in chunks of {max_seq_length}") if not data_args.streaming and training_args.do_train else (tokenized_datasets.map(group_texts, batched=True) if data_args.streaming else tokenized_datasets.map(group_texts, batched=True))

    train_dataset = tokenized_datasets["train"] if training_args.do_train else None
    eval_dataset = tokenized_datasets["validation"] if training_args.do_eval and ("validation" in tokenized_datasets) else None
    train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples))) if data_args.max_train_samples is not None and training_args.do_train else train_dataset
    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples))) if eval_dataset and data_args.max_eval_samples is not None else eval_dataset

    preprocess_logits_for_metrics = lambda logits: (logits[0] if isinstance(logits, tuple) else logits).argmax(dim=-1)
    labels = lambda pred: pred.reshape(-1)
    mask = lambda lbl: lbl[lbl != -100]
    return metric.compute(predictions=pred[mask], references=lbl[mask]) if (pred := preprocess_logits_for_metrics(eval_preds)) and (lbl := labels(eval_preds)) else eval_preds

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability, pad_to_multiple_of=8 if (data_args.line_by_line and training_args.fp16) else None)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train else None, eval_dataset=eval_dataset if training_args.do_eval else None, processing_class=tokenizer, data_collator=data_collator, compute_metrics=(lambda ex: metric.compute(predictions=pred[mask], references=lbl[mask])) if training_args.do_eval and not is_torch_xla_available() else None, preprocess_logits_for_metrics=(preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None))

    checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint is not None else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint) if training_args.do_train else None
    trainer.save_model() if training_args.do_train else None
    metrics = train_result.metrics if training_args.do_train else {}
    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics) if training_args.do_train else None
    trainer.save_metrics("train", metrics) if training_args.do_train else None
    trainer.save_state() if training_args.do_train else None

    metrics = trainer.evaluate() if training_args.do_eval else {}
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset)) if eval_dataset else 0
    try: perplexity = math.exp(metrics["eval_loss"]) except OverflowError: perplexity = float("inf")
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics) if training_args.do_eval else None
    trainer.save_metrics("eval", metrics) if training_args.do_eval else None

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name:
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset_args"] = data_args.dataset_config_name if data_args.dataset_config_name else None
        kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}" if data_args.dataset_config_name else data_args.dataset_name

    trainer.push_to_hub(**kwargs) if training_args.push_to_hub else (trainer.create_model_card(**kwargs))
    main()

def _mp_fn(index): main()

if __name__ == "__main__": main()