# -*- coding: utf-8 -*-
import dataclasses

import torch
import transformers as tf


@dataclasses.dataclass
class ModelArguments:
    text_model_name_or_path: str
    audio_model_name_or_path: str
    num_classes: int = dataclasses.field(default=None)


def freeze_params(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


# XXX: This doesn't need to make the tokenizers.
# TODO: https://stackoverflow.com/questions/73948214/how-to-convert-a-pytorch-nn-module-into-a-huggingface-pretrainedmodel-object
class MultimodalClassifier(torch.nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.config = config
        # Load the text model.
        #  self.text_tokenizer = tf.AutoTokenizer.from_pretrained(
        #      config.text_model_name_or_path
        #  )
        self.text_model = tf.AutoModel.from_pretrained(config.text_model_name_or_path)
        # Load the audio model.
        #  self.audio_feature_extractor = tf.AutoFeatureExtractor.from_pretrained(
        #      config.audio_model_name_or_path
        #  )
        self.audio_model = tf.AutoModel.from_pretrained(config.audio_model_name_or_path)
        # Initialize classification head.
        self.classifier_proj_size = (
            self.text_model.config.hidden_size + self.audio_model.config.hidden_size
        )
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(
                self.text_model.config.hidden_size + self.audio_model.config.hidden_size,
                self.classifier_proj_size,
            ),  # Dense projection layer.
            torch.nn.ReLU(),  # Activation.
            torch.nn.Linear(self.classifier_proj_size, config.num_classes)  # Classifier.
        )

    def forward(self, text_input_ids, text_attention_mask, audio_input_values, labels, **kwargs):
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/t5/modeling_t5.py#L2053
        text_decoder_input_ids = self.text_model._shift_right(text_input_ids)
        text_features = self.text_model(
            input_ids=text_input_ids,
            decoder_input_ids=text_decoder_input_ids,
            attention_mask=text_attention_mask
        )
        audio_features = self.audio_model(audio_input_values)
        # Max Pooling. TODO: support more pooling options.
        text_pooled = torch.max(text_features.last_hidden_state, dim=1).values
        audio_pooled = torch.max(audio_features.last_hidden_state, dim=1).values
        # Classification logits.
        fusion_features = torch.cat([text_pooled, audio_pooled], dim=1)
        logits = self.classification_head(fusion_features)
        # Compute loss.
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
        # Return.
        return tf.modeling_outputs.SequenceClassifierOutput(loss=loss, logits=logits)
