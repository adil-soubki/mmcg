# -*- coding: utf-8 -*-
import dataclasses
from typing import Literal, Optional

import torch
import transformers as tf

from ..data.commitment_bank import load_opensmile


PoolerType = Literal["max", "mean", "sum"]  # XXX: Unused currently.
@dataclasses.dataclass
class ModelArguments:
    text_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    audio_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    use_opensmile_features: bool = dataclasses.field(default=False)
    num_labels: int = dataclasses.field(default=None)
    # XXX: Unused currently
    text_pooler_type: Optional[PoolerType] = dataclasses.field(default="max")
    audio_pooler_type: Optional[PoolerType] = dataclasses.field(default="max")


# XXX: Unused currently.
def freeze_params(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


# XXX: Unused currently.
def pooler(features: torch.Tensor, dim: int, pooler_type: PoolerType) -> torch.Tensor:
    if not features.numel():
        return features  # Nothing to pool.
    if pooler_type == "max":
        pooler = lambda t, d: torch.max(t, dim=d).values
    elif pooler_type == "mean":
        pooler = torch.mean
    elif pooler_type == "sum":
        pooler = torch.sum
    else:
        raise ValueError(f"unknown pooler_type: {pooler_type}")
    return pooler(features, dim=dim)


# TODO: https://stackoverflow.com/questions/73948214/how-to-convert-a-pytorch-nn-module-into-a-huggingface-pretrainedmodel-object
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
        # Load the audio model.
        self.audio_model = (
            tf.AutoModel.from_pretrained(config.audio_model_name_or_path)
            if config.audio_model_name_or_path
            else None
        )
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
            torch.nn.ReLU(),  # Activation. TODO: Dropout?
            torch.nn.Linear(self.classifier_proj_size, config.num_labels)  # Classifier.
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
            text_decoder_input_ids = self.text_model._shift_right(text_input_ids)
            text_features = self.text_model(
                input_ids=text_input_ids,
                decoder_input_ids=text_decoder_input_ids,
                attention_mask=text_attention_mask
            ).last_hidden_state
        audio_features = torch.tensor([]).to(device)
        if self.audio_model:
            audio_features = self.audio_model(audio_input_values).last_hidden_state
        if not self.config.use_opensmile_features:
            opensmile_features = torch.tensor([]).to(device)
        # Max Pooling. TODO: support more pooling options.
        text_pooled = (
            torch.max(text_features, dim=1).values
            if text_features.numel()
            else text_features
        )
        audio_pooled = (
            torch.max(audio_features, dim=1).values
            if audio_features.numel()
            else audio_features
        )
        # Classification logits.
        fusion_features = torch.cat([
            text_pooled, audio_pooled, opensmile_features
        ], dim=1)
        logits = self.classification_head(fusion_features)
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
