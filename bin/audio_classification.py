#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import datasets
import evaluate
import numpy as np
import transformers as tf

from src.core.context import Context
from src.core.app import harness


def main(ctx: Context) -> None:
    args = ctx.parser.parse_args()
    # XXX: See this issue on Whisper issue ("openai/whisper-base").
    #   https://github.com/huggingface/transformers/issues/25744
    audio_model_name_or_path = "facebook/wav2vec2-base"
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
    # Preprocess training data.
    feature_extractor = tf.AutoFeatureExtractor.from_pretrained(audio_model_name_or_path)
    minds = minds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    def preprocess_fn(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=16_000,
            truncation=True
        )
        return inputs
    encoded_minds = minds.map(preprocess_fn, remove_columns="audio", batched=True)
    encoded_minds = encoded_minds.rename_column("intent_class", "label")
    # Model training.
    model = tf.AutoModelForAudioClassification.from_pretrained(
        audio_model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label
    )
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return evaluate.load("accuracy").compute(
            predictions=predictions,
            references=eval_pred.label_ids
        )
    training_args = tf.TrainingArguments(
        output_dir="./outputs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_minds["train"],
        eval_dataset=encoded_minds["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    harness(main)
