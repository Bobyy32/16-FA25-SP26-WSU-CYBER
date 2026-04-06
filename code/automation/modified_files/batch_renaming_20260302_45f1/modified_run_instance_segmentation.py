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

from src.utils import setup_logging, find_last_checkpoint


class EvaluationHandler:
    """Handles evaluation metrics computation for instance segmentation tasks."""
    
    def __init__(self, image_processor, id2label, threshold=0.0):
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = transformers.Metric()

    def postprocess_target_batch(self, target_batch):
        """Convert targets to required format for metric computation."""
        return [
            {"masks": target["masks"], "labels": target["labels"]}
            for target in target_batch
        ]

    def get_target_sizes(self, post_processed_targets):
        """Extract target sizes from processed targets."""
        return [
            target["masks"].shape[1:] for target in post_processed_targets
        ]

    def postprocess_prediction_batch(self, prediction_batch, target_sizes):
        """Convert predictions to required format for metric computation."""
        results = []
        for pred, target_size in zip(prediction_batch, target_sizes):
            if pred["segments_info"]:
                results.append({
                    "masks": pred["segmentation"],
                    "labels": [x["label_id"] for x in pred["segments_info"]],
                    "scores": [x["score"] for x in pred["segments_info"]],
                })
            else:
                results.append({
                    "masks": pred["segmentation"],
                    "labels": [],
                    "scores": [],
                })
        return results

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, compute_result: bool = False) -> Mapping[str, float]:
        """Compute metrics for evaluation results."""
        prediction_batch = evaluation_results.predictions
        target_batch = evaluation_results.label_ids

        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(prediction_batch, target_sizes)

        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return {}

        metrics = self.metric.compute()

        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
        return metrics


def augment_and_transform_batch(dataset_item, transform, image_processor):
    """Apply transformations to dataset items."""
    # This function would be implemented to apply transformations
    # to images and annotations in the dataset
    pass


def main():
    # Parse arguments
    parser = HfArgumentParser([Arguments, TrainingArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    # Configure training arguments
    training_args.eval_do_concat_batches = False
    training_args.batch_eval_metrics = True
    training_args.remove_unused_columns = False

    # Setup logging
    setup_logging(training_args)

    # Find last checkpoint
    checkpoint = find_last_checkpoint(training_args)

    # Load dataset
    dataset = load_dataset(args.dataset_name, trust_remote_code=args.trust_remote_code)
    label2id = dataset["train"][0]["semantic_class_to_id"]

    if args.do_reduce_labels:
        label2id = {name: idx for name, idx in label2id.items() if idx != 0}
        label2id = {name: idx - 1 for name, idx in label2id.items()}

    id2label = {v: k for k, v in label2id.items()}

    # Load model and processor
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

    # Define transforms
    train_augment_and_transform = [
        transformers.transforms.HorizontalFlip(p=0.5),
        transformers.transforms.RandomBrightnessContrast(p=0.5),
        transformers.transforms.HueSaturationValue(p=0.1),
    ]
    validation_transform = [transformers.transforms.NoOp()]

    # Apply transforms to datasets
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)

    # Initialize trainer
    compute_metrics = EvaluationHandler(image_processor=image_processor, id2label=id2label, threshold=0.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=lambda x: x,  # Custom collator needed
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Model card
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