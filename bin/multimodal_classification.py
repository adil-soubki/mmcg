#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import dataclasses
import os
import sys
from typing import Optional

import datasets
import evaluate
import numpy as np
import transformers as tf

from src.core import nvidia
from src.core.context import Context
from src.core.app import harness
from src.models.multimodal_classifier import MultimodalClassifier, ModelArguments


@dataclasses.dataclass
class DataArguments:
    do_regression: bool = dataclasses.field(
        default=None,
        metadata={
            "help": (
                "Whether to do regression instead of classification. If None, "
                "will be inferred from the dataset."
            )
        },
    )
    max_seq_length: int = dataclasses.field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
		"Sequences longer than this will be truncated, sequences shorter "
		"will be padded."
            )
        },
    )
    train_file: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )


def main(ctx: Context) -> None:
    # Parse arguments.
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, tf.TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    ctx.log.info(f"Training parameters {training_args}")
    ctx.log.info(f"Data parameters {data_args}")
    ctx.log.info(f"Model parameters {model_args}")
    # Select lowest memory GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(nvidia.best_gpu())
    ctx.log.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    # Set seed before initializing model.
    tf.set_seed(training_args.seed)
    # Load training data.
    minds = datasets.load_dataset(
        "PolyAI/minds14",
        name="en-US",
        split="train",
        trust_remote_code=True
    )
    minds = minds.train_test_split(test_size=0.2)
    labels = minds["train"].features["intent_class"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    model_args.num_classes = model_args.num_classes or len(labels)
    # Preprocess training data.
    feature_extractor = tf.AutoFeatureExtractor.from_pretrained(
        model_args.audio_model_name_or_path
    ) if model_args.audio_model_name_or_path else None
    tokenizer = tf.AutoTokenizer.from_pretrained(
        model_args.text_model_name_or_path
    ) if model_args.text_model_name_or_path else None
    assert not tokenizer or (tokenizer.model_max_length >= data_args.max_seq_length)
    minds = minds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    def preprocess_fn(examples):
        dummy = [[0]] * len(examples["intent_class"])
        # Audio processing.
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=getattr(feature_extractor, "sampling_rate", 16_000),
            max_length=16_000,
            truncation=True
        ) if feature_extractor else {"input_values": dummy}
        # Text processing.
        inputs |= tokenizer(
            examples["english_transcription"],
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True
        ) if tokenizer else {"input_ids": dummy, "attention_mask": dummy}
        return inputs
    minds = minds.map(preprocess_fn, batched=True, batch_size=16)
    minds = minds.rename_columns({
        "input_ids": "text_input_ids",
        "attention_mask": "text_attention_mask",
        "input_values": "audio_input_values",
        "intent_class": "label",
    })
    train_dataset, eval_dataset = minds["train"], minds["test"]
    # Model training.
    model = MultimodalClassifier(model_args)
    def compute_metrics(eval_pred: tf.EvalPrediction):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        # Save predictions to file.
        cols = ["transcription", "english_transcription", "label", "lang_id"]
        pdf = minds["test"].to_pandas()[cols].assign(pred=predictions)
        assert (pdf.label == eval_pred.label_ids).all()
        pdf.to_csv(os.path.join(training_args.output_dir, "preds.csv"))
        # Return metrics.
        return evaluate.load("accuracy").compute(
            predictions=predictions,
            references=eval_pred.label_ids
        )
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    harness(main)
