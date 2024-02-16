# -*- coding: utf-8 -*-
import torch
import transformers as tf


class MultimodalClassifierConfig:
    text_model_name_or_path: str
    audio_model_name_or_path: str
    num_classes: str


class MultimodalClassifier(torch.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Load the text model.
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            config.text_model_name_or_path
        )
        self.text_model = AutoModel.from_pretrained(config.model_name_or_path)
        # Load the audio model.
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.audio_model_name_or_path
        )
        self.audio_model = AutoModel.from_pretrained(config.audio_model_name_or_path)
        # Initialize classification head.
        self.classifier_proj_size = (
            self.text_model.config.hidden_size + self.audio_model.config.hidden_size
        )
        self.classification_head = torch.Sequential(
            torch.Linear(
                self.text_model.config.hidden_size + self.audio_model.config.hidden_size,
                self.classifier_proj_size,
            ),  # Dense projection layer.
            torch.ReLU(),  # Activation.
            torch.Linear(self.classifier_proj_size, config.num_classes)  # Classifier.
        )

    def forward(self, batch):
        encodings = []
        for data in batch:
            encodings.append(
                torch.cat(
                    # Use whole text model.
                    self.text_model(data["text"]),
                    # Just use audio encoder (like WhisperForAudioClassification).
                    self.audio_model.encoder(data["audio"]),
                    dim=0
                )
            )
        return self.classification_head(encodings)  # Logits.
