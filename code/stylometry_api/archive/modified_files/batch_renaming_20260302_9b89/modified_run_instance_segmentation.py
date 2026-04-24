import os
import sys
from functools import partial
from typing import Mapping, Optional

import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from datasets import load_dataset

from . import logger


class Evaluator:
    def __init__(self, image_processor, id2label, threshold=0.0):
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = transformers.Metric()

    def postprocess_target_batch(self, target_batch):
        return [dict(masks=x[0], labels=x[1]) for x in target_batch]

    def get_target_sizes(self, post_processed_targets):
        return [x["masks"].shape[1:] for x in post_processed_targets]

    def postprocess_prediction_batch(self, prediction_batch, target_sizes):
        return [
            {
                "masks": x["segmentation"].to(dtype=torch.bool),
                "labels": torch.tensor([seg["label_id"] for seg in x["segments_info"]]),
                "scores": torch.tensor([seg["score"] for seg in x["segments_info"]]),
            }
            if x["segments_info"]
            else {
                "masks": torch.zeros([0, *size], dtype=torch.bool),
                "labels": torch.tensor([]),
                "scores": torch.tensor([]),
            }
            for x, size in zip(prediction_batch, target_sizes)
        ]

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, compute_result: bool = False):
        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)

        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(prediction_batch, target_sizes)

        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return

        metrics = self.metric.compute()

        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        self.metric.reset()

        return metrics


def nested_cpu(x):
    return x.cpu() if hasattr(x, 'cpu') else [nested_cpu(item) for item in x]


def setup_logging(training_args: TrainingArguments) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def find_last_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    return checkpoint


def main():
    parser = HfArgumentParser([Arguments, TrainingArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    training_args.eval_do_concat_batches = False
    training_args.batch_eval_metrics = True
    training_args.remove_unused_columns = False

    setup_logging(training_args)
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    checkpoint = find_last_checkpoint(training_args)

    dataset = load_dataset(args.dataset_name, trust_remote_code=args.trust_remote_code)

    label2id = dataset["train"][0]["semantic_class_to_id"]

    if args.do_reduce_labels:
        label2id = {name: idx for name, idx in label2id.items() if idx != 0}
        label2id = {name: idx - 1 for name, idx in label2id.items()}

    id2label = {v: k for k, v in label2id.items()}

    model = AutoModelForUniversalSegmentation.from_pretrained(
        args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        token=args.token,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        do_resize=True,
        size={"height": args.image_height, "width": args.image_width},
        do_reduce_labels=args.do_reduce_labels,
        reduce_labels=args.do_reduce_labels,
        token=args.token,
    )

    train_augment_and_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ])
    validation_transform = A.Compose([A.NoOp()])

    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)

    compute_metrics = Evaluator(image_processor=image_processor, id2label=id2label, threshold=0.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    kwargs = {
        "finetuned_from": args.model_name_or_path,
        "dataset": args.dataset_name,
        "tags": ["image-segmentation", "instance-segmentation", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()