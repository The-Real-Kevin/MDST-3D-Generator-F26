"""
Text -> city-parameter model.

Architecture (deliberately small, laptop-CPU-friendly):

    sentence text
        -> frozen pretrained sentence embedding (all-MiniLM-L6-v2, ~80MB)
           NOT trained -- just a fixed feature extractor
        -> small shared MLP trunk (this + the heads are what actually gets trained)
        -> one output head per parameter in PARAM_SCHEMA:
             - categorical params (terrain, city_archetype): softmax classification head
             - continuous params (density, relief, ...): sigmoid-scaled regression head,
               scaled into that parameter's [low, high] range
             - binary params (green_override): single sigmoid logit

Why this shape: the embedding model already knows what "densely packed" or
"seaside" mean in general English -- we don't need to (and can't, on this
timeline) teach a model English from scratch. The only thing that needs
training is the small trunk+heads that map "meaning of the sentence" to
"this specific generator's 10 numbers". That's a few thousand parameters,
not billions, which is exactly why this trains in minutes on a CPU instead
of needing a GPU cluster.

Independent degrees of freedom: every parameter gets its own head reading
from the same shared trunk output, so each one is free to vary on its own
-- the model isn't forced to move density and relief together just because
they came from the same sentence.

Getting sentence embeddings (frozen, not trained) is handled separately in
embedders.py -- this file only defines the trainable trunk+heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from text2city.schema import PARAM_SCHEMA

EMBEDDING_DIM_DEFAULT = 384  # matches all-MiniLM-L6-v2; override per embedder used


class TextToCityParams(nn.Module):
    """
    Shared trunk + one head per parameter. Call get_sentence_embedder()
    separately (it's frozen, not part of this module's trainable weights)
    and feed its output in as `embedding`.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM_DEFAULT, hidden_dim: int = 128):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # one head per parameter, keyed by name so training/inference code
        # can look them up without relying on dict ordering
        self.heads = nn.ModuleDict()
        for spec in PARAM_SCHEMA:
            if spec.kind == "categorical":
                self.heads[spec.name] = nn.Linear(hidden_dim, len(spec.choices))
            elif spec.kind == "continuous":
                self.heads[spec.name] = nn.Linear(hidden_dim, 1)
            elif spec.kind == "binary":
                self.heads[spec.name] = nn.Linear(hidden_dim, 1)
            else:
                raise ValueError(f"unknown parameter kind: {spec.kind}")

    def forward(self, embedding: torch.Tensor) -> dict:
        """
        embedding: (batch, embedding_dim) frozen sentence embeddings.
        Returns a dict of raw head outputs (logits for categorical/binary,
        unscaled values for continuous) -- see decode_outputs() for turning
        these into actual usable parameter values.
        """
        features = self.trunk(embedding)
        return {name: head(features) for name, head in self.heads.items()}


def decode_outputs(raw_outputs: dict) -> dict:
    """
    Turns raw model outputs into actual parameter values:
        categorical -> the argmax class name
        continuous  -> sigmoid(raw) rescaled into [low, high]
        binary      -> sigmoid(raw) > 0.5
    Works on a single example's outputs (no batch dimension) for simplicity;
    see decode_batch() for evaluating many examples at once.
    """
    result = {}
    schema_by_name = {p.name: p for p in PARAM_SCHEMA}

    for name, raw in raw_outputs.items():
        spec = schema_by_name[name]
        if spec.kind == "categorical":
            idx = torch.argmax(raw).item()
            result[name] = spec.choices[idx]
        elif spec.kind == "continuous":
            low, high = spec.value_range
            frac = torch.sigmoid(raw).item()
            result[name] = low + frac * (high - low)
        elif spec.kind == "binary":
            result[name] = torch.sigmoid(raw).item() > 0.5
    return result


def decode_batch(raw_outputs: dict, index: int) -> dict:
    """Same as decode_outputs, but pulls one row out of a batched forward() output."""
    single = {name: raw[index] for name, raw in raw_outputs.items()}
    return decode_outputs(single)
