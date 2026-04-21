"""Discogs-Effnet via Essentia — embedding (1280-dim) + Discogs genre labels.

Model files are not bundled. Download from
https://essentia.upf.edu/models.html#discogs-effnet and pass paths (or set
SOUNDLIB_DISCOGS_EMBED / SOUNDLIB_DISCOGS_CLASSIFIER env vars):

    discogs-effnet-bs64-1.pb                 (embedding model)
    genre_discogs400-discogs-effnet-1.pb     (400-way genre classifier)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from soundlib.embeddings.base import Backend, EmbeddingResult


DISCOGS_SR = 16000
DISCOGS_DIM = 1280


class DiscogsEffnetBackend(Backend):
    name = "discogs-effnet"
    sample_rate = DISCOGS_SR
    dim = DISCOGS_DIM

    def __init__(
        self,
        embedding_model: str | None = None,
        classifier_model: str | None = None,
        labels_json: str | None = None,
    ):
        try:
            from essentia.standard import (
                TensorflowPredictEffnetDiscogs,
                TensorflowPredict2D,
            )
        except ImportError as e:
            raise ImportError(
                "essentia-tensorflow is required for the discogs-effnet backend. "
                "Install with: pip install soundlib[essentia]"
            ) from e

        emb_path = embedding_model or os.environ.get("SOUNDLIB_DISCOGS_EMBED")
        clf_path = classifier_model or os.environ.get("SOUNDLIB_DISCOGS_CLASSIFIER")
        if not emb_path:
            raise FileNotFoundError(
                "discogs-effnet embedding model path not provided. "
                "Pass embedding_model= or set SOUNDLIB_DISCOGS_EMBED."
            )
        if not Path(emb_path).exists():
            raise FileNotFoundError(f"embedding model not found: {emb_path}")

        self._embed_net = TensorflowPredictEffnetDiscogs(
            graphFilename=emb_path, output="PartitionedCall:1"
        )
        self._classifier = None
        self._labels: list[str] = []
        if clf_path:
            if not Path(clf_path).exists():
                raise FileNotFoundError(f"classifier model not found: {clf_path}")
            self._classifier = TensorflowPredict2D(
                graphFilename=clf_path,
                input="serving_default_model_Placeholder",
                output="PartitionedCall:0",
            )
            self._labels = _load_labels(labels_json, clf_path)

    def embed(self, audio: np.ndarray) -> EmbeddingResult:
        audio = np.asarray(audio, dtype=np.float32).ravel()
        # Essentia returns (N, 1280); average across time for a single track vector.
        frames = self._embed_net(audio)
        embedding = frames.mean(axis=0).astype(np.float32)

        genres: list[tuple[str, float]] = []
        if self._classifier is not None and self._labels:
            probs = self._classifier(frames).mean(axis=0)
            order = np.argsort(-probs)
            genres = [(self._labels[i], float(probs[i])) for i in order[:20]]

        return EmbeddingResult(embedding=embedding, genres=genres)


def _load_labels(labels_json: str | None, classifier_path: str) -> list[str]:
    if labels_json and Path(labels_json).exists():
        return _read_labels(labels_json)
    # Essentia ships labels alongside the model as a .json sidecar.
    sidecar = Path(classifier_path).with_suffix(".json")
    if sidecar.exists():
        return _read_labels(sidecar)
    return []


def _read_labels(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    for key in ("classes", "labels"):
        if key in payload:
            return [str(x) for x in payload[key]]
    raise ValueError(f"could not find class labels in {path}")
