# -*- coding: utf-8 -*-
import dataclasses
from typing import Literal, Optional

import torch
import transformers as tf

from ..data.commitment_bank import load_opensmile


PoolerType = Literal["max", "mean", "sum"]
FusionStrategy = Literal["early", "late"]
@dataclasses.dataclass
class ModelArguments:
    text_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    audio_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    use_opensmile_features: bool = dataclasses.field(default=False)
    num_labels: int = dataclasses.field(default=None)
    text_pooler_type: Optional[PoolerType] = dataclasses.field(default="max")
    audio_pooler_type: Optional[PoolerType] = dataclasses.field(default="max")
    freeze_text_model: bool = dataclasses.field(default=False)
    freeze_audio_model: bool = dataclasses.field(default=False)
    fusion_strategy: FusionStrategy = dataclasses.field(default="early")


def freeze_params(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def pooler(features: torch.Tensor, dim: int, pooler_type: PoolerType) -> torch.Tensor:
    if not features.numel():
        return features  # Nothing to pool.
    if pooler_type == "max":
        pool_fn = lambda t, dim: torch.max(t, dim=dim).values
    elif pooler_type == "mean":
        pool_fn = torch.mean  # TODO: version that drops padded columns.
    elif pooler_type == "sum":
        pool_fn = torch.sum
    else:
        raise ValueError(f"unknown pooler_type: {pooler_type}")
    return pool_fn(features, dim=dim)


def classification_head(
    input_size: int, proj_size: int, output_size: int
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, proj_size),  # Dense projection layer.
        #  torch.nn.LayerNorm(proj_size),        # XXX: Make optional?
        torch.nn.ReLU(),                         # Activation. TODO: Dropout?
        torch.nn.Linear(proj_size, output_size)  # Classifier.
    )


class MultimodalClassifier(torch.nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.config = config
        # Load the text model.
        self.text_model = (
            tf.AutoModel.from_pretrained(config.text_model_name_or_path)
            if config.text_model_name_or_path
            else None
        )
        if self.text_model and self.config.freeze_text_model:
            freeze_params(self.text_model)
        # Load the audio model.
        self.audio_model = (
            tf.AutoModel.from_pretrained(config.audio_model_name_or_path)
            if config.audio_model_name_or_path
            else None
        )
        if self.audio_model and self.config.freeze_audio_model:
            freeze_params(self.audio_model)
        # Throw if neither is given.
        if (
            not self.text_model and
            not self.audio_model and
            not self.config.use_opensmile_features
        ):
            raise ValueError("No text or audio model(s) specified.")
        # Initialize classification head.
        text_hidden_size = self.text_model.config.hidden_size if self.text_model else 0
        audio_hidden_size = self.audio_model.config.hidden_size if self.audio_model else 0
        opensmile_hidden_size = (
            load_opensmile().opensmile_features[0].shape[0]
            if self.config.use_opensmile_features
            else 0
        )
        self.classifier_proj_size = (
            text_hidden_size + audio_hidden_size + opensmile_hidden_size
        )
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(
                text_hidden_size + audio_hidden_size + opensmile_hidden_size,
                self.classifier_proj_size,
            ),  # Dense projection layer.
            #  torch.nn.LayerNorm(self.classifier_proj_size),  # XXX: Make optional?
            torch.nn.ReLU(),  # Activation. TODO: Dropout?
            torch.nn.Linear(self.classifier_proj_size, config.num_labels)  # Classifier.
        )
        # Initialize late fusion classification heads.
        self.text_classification_head = classification_head(
            text_hidden_size, text_hidden_size, config.num_labels
        )
        self.audio_classification_head = classification_head(
            audio_hidden_size, audio_hidden_size, config.num_labels
        )
        self.opensmile_classification_head = classification_head(
            opensmile_hidden_size, opensmile_hidden_size, config.num_labels
        )

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        audio_input_values,
        opensmile_features,
        labels,
        **kwargs
    ):
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/t5/modeling_t5.py#L2053
        device = self.classification_head[0].weight.device
        text_features = torch.tensor([]).to(device)
        if self.text_model:
            text_features = self.text_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            ).last_hidden_state
        audio_features = torch.tensor([]).to(device)
        if self.audio_model and self.audio_model.__class__.__name__ == "WhisperModel":
            audio_features = self.audio_model.encoder(audio_input_values).last_hidden_state
        elif self.audio_model:
            audio_features = self.audio_model(audio_input_values).last_hidden_state
        if not self.config.use_opensmile_features:
            opensmile_features = torch.tensor([]).to(device)
        # Pooling.
        text_pooled = pooler(
            text_features, dim=1, pooler_type=self.config.text_pooler_type
        )
        audio_pooled = pooler(
            audio_features, dim=1, pooler_type=self.config.audio_pooler_type
        )
        # Classification logits.
        if self.config.fusion_strategy == "early":
            fusion_features = torch.cat([
                text_pooled, audio_pooled, opensmile_features
            ], dim=1)
            logits = self.classification_head(fusion_features)
        elif self.config.fusion_strategy == "late":
            text_logits = torch.tensor([]).to(device)
            if self.text_model:
                text_logits = self.text_classification_head(text_pooled)
            audio_logits = torch.tensor([]).to(device)
            if self.audio_model:
                audio_logits = self.audio_classification_head(audio_pooled)
            opensmile_logits = torch.tensor([]).to(device)
            if self.config.use_opensmile_features:
                opensmile_logits = self.opensmile_classification_head(opensmile_features)
            logits = torch.cat([text_logits, audio_logits, opensmile_logits]).mean()
        else:
            raise ValueError
        # Compute loss.
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # TODO: Consider weighting? label_smoothing?
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # Return.
        return tf.modeling_outputs.SequenceClassifierOutput(loss=loss, logits=logits)
