#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import dataclasses
import os
import sys
from contextlib import suppress
from copy import copy

import datasets
import evaluate
import numpy as np
import pandas as pd
import transformers as tf

from src.core.context import Context
from src.core.app import harness
from src.data import commitment_bank
from src.core.df import update
from src.models.multimodal_classifier import MultimodalClassifier, ModelArguments


@dataclasses.dataclass
class DataArguments:
    data_num_folds: int
    data_fold: int = dataclasses.field(default=None)
    do_regression: bool = dataclasses.field(default=False)
    metric_for_classification: str = dataclasses.field(default="accuracy")
    metric_for_regression: str = dataclasses.field(default="mae")
    text_max_length: int = dataclasses.field(
        default=128,
        metadata={
            "help": (
                "The maximum total text input sequence length after tokenization. "
		"Sequences longer than this will be truncated, sequences shorter "
		"will be padded."
            )
        },
    )
    audio_max_length: int = dataclasses.field(
        default=16,
        metadata={
            "help": (
                "The maximum audio length in seconds for feature extraction. "
		"Sequences longer than this will be truncated, sequences shorter "
		"will be padded."
            )
        },
    )


def update_metrics(
    preds: list,
    refs: list,
    trainer: tf.Trainer,
    raw_args: dict,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: tf.TrainingArguments
):
    metric = (
        data_args.metric_for_regression
        if data_args.do_regression
        else data_args.metric_for_classification
    )
    # Read the current configuration.
    raw_args["data_fold"] = data_args.data_fold
    raw_args["current_epoch"] = trainer.state.epoch
    output_path = os.path.join(
        os.path.dirname(raw_args["output_dir"]), "all_results.csv"
    )
    # Read in the current results.
    current = pd.DataFrame()
    with suppress(FileNotFoundError):
        current = pd.read_csv(output_path)
    # Compute the new results.
    results = evaluate.combine([metric, "pearsonr"]).compute(
        predictions=preds, references=refs
    )
    new = pd.DataFrame([raw_args | results])
    new["last_modified"] = pd.Timestamp.now()
    # Normalize column names (fill new ones with None).
    if current.empty:
        current = pd.DataFrame(columns=new.columns)
    current = current.assign(**{
        c: None for c in set(new.columns) - set(current.columns)
    })
    new = new.assign(**{c: None for c in set(current.columns) - set(new.columns)})
    # Update the results file.
    on_keys = list(
        set(current.columns) - (set(results.keys()) | {"last_modified", "current_epoch"})
    )
    update(current, new, on=on_keys).to_csv(output_path, index=False)


def run(
    ctx: Context,
    raw_args: dict,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: tf.TrainingArguments
) -> None:
    # Make a directory per fold.
    training_args.output_dir = os.path.join(
        training_args.output_dir, f"fold_{data_args.data_fold}"
    )
    ctx.log.info(f"Training parameters {training_args}")
    ctx.log.info(f"Data parameters {data_args}")
    ctx.log.info(f"Model parameters {model_args}")
    # Set seed before initializing model.
    tf.set_seed(training_args.seed)
    # Configure for regression if needed.
    metric = (
        data_args.metric_for_regression
        if data_args.do_regression
        else data_args.metric_for_classification
    )
    # XXX: Currently not needed.
    training_args.greater_is_better = metric not in ("loss", "eval_loss", "mse", "mae")
    # Load training data.
    data = commitment_bank.load_kfold(
        num_labels=model_args.num_labels,
        fold=data_args.data_fold,
        k=data_args.data_num_folds,
        seed=training_args.data_seed
    )
    if data_args.do_regression:
        data = data.remove_columns("cb_val").rename_column("cb_val_float", "cb_val")
        model_args.num_labels = 1  # NOTE: Just used to stratify.
    # Preprocess training data.
    feature_extractor = tf.AutoFeatureExtractor.from_pretrained(
        model_args.audio_model_name_or_path
    ) if model_args.audio_model_name_or_path else None
    tokenizer = tf.AutoTokenizer.from_pretrained(
        model_args.text_model_name_or_path
    ) if model_args.text_model_name_or_path else None
    assert not tokenizer or (tokenizer.model_max_length >= data_args.text_max_length)
    data = data.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    def preprocess_fn(examples):
        dummy = [[0]] * len(examples[list(examples.keys())[0]])
        # Audio processing.
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=getattr(feature_extractor, "sampling_rate", 16_000),
            padding="max_length",
            max_length=data_args.audio_max_length * 16_000,
            truncation=True
        ) if feature_extractor else {"input_values": dummy}
        # Text processing.
        inputs |= tokenizer(
            examples["cb_target"],
            padding="max_length",
            max_length=data_args.text_max_length,
            truncation=True
        ) if tokenizer else {"input_ids": dummy, "attention_mask": dummy}
        return inputs
    data = data.map(preprocess_fn, batched=True, batch_size=16)
    data = data.rename_columns({
        "input_ids": "text_input_ids",
        "attention_mask": "text_attention_mask",
        "input_values": "audio_input_values",
        "cb_val": "label",
    })
    train_dataset, eval_dataset = data["train"], data["test"]
    # Model training.
    model = MultimodalClassifier(model_args)
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    def compute_metrics(eval_pred: tf.EvalPrediction):
        if data_args.do_regression:
            predictions = np.squeeze(eval_pred.predictions)
        else:
            predictions = np.argmax(eval_pred.predictions, axis=1)
        # Save predictions to file.
        cols = ["number", "clip_start", "clip_end", "cb_target", "label"]
        pdf = eval_dataset.to_pandas()[cols].assign(pred=predictions)
        assert np.allclose(pdf.label, eval_pred.label_ids)
        pdf.to_csv(os.path.join(training_args.output_dir, "preds.csv"))
        # Update aggregated evaluation results.
        update_metrics(
            predictions, eval_pred.label_ids, trainer,
            raw_args, model_args, data_args, training_args
        )
        # Return metrics.
        return evaluate.combine([metric, "pearsonr"]).compute(
            predictions=predictions,
            references=eval_pred.label_ids
        )
    trainer.compute_metrics = compute_metrics
    trainer.train()
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def main(ctx: Context) -> None:
    # Parse arguments.
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, tf.TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
        raw_args = pd.read_json(
            os.path.abspath(sys.argv[1]), orient="index"
        )[0].to_dict()
    else:
        parser.error("No configuration passed")
    # Run the training loop.
    if data_args.data_fold is not None:
        return run(ctx, model_args, data_args, training_args)
    for fold in range(data_args.data_num_folds):
        data_args.data_fold = fold
        run(ctx, raw_args, copy(model_args), copy(data_args), copy(training_args))


if __name__ == "__main__":
    harness(main)
