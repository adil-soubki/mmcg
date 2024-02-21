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
    )
    tokenizer = tf.AutoTokenizer.from_pretrained(
        model_args.text_model_name_or_path
    )
    assert tokenizer.model_max_length >= data_args.max_seq_length
    minds = minds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    def preprocess_fn(examples):
        # Audio processing.
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=16_000,
            truncation=True
        )
        # Text processing.
        inputs |= tokenizer(
            examples["english_transcription"],
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True
        )
        return inputs
    minds = minds.map(preprocess_fn, batched=True, batch_size=16)
    minds = minds.rename_columns({
        "input_ids": "text_input_ids",
        "attention_mask": "text_attention_mask",
        "input_values": "audio_input_values",
        "intent_class": "label",
    })
    # Model training.
    model = MultimodalClassifier(model_args)
    #  model = tf.AutoModel.from_pretrained(model_args.text_model_name_or_path)
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return evaluate.load("accuracy").compute(
            predictions=predictions,
            references=eval_pred.label_ids
        )
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=minds["train"],
        eval_dataset=minds["test"],
        #  tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    harness(main)
