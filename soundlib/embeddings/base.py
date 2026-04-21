"""Backend ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmbeddingResult:
    embedding: np.ndarray                       # 1-D float32
    genres: list[tuple[str, float]] = field(default_factory=list)  # (label, prob), sorted desc
    extra: dict = field(default_factory=dict)


class Backend(ABC):
    name: str
    sample_rate: int
    dim: int

    @abstractmethod
    def embed(self, audio: np.ndarray) -> EmbeddingResult:
        """Embed a mono float32 array sampled at self.sample_rate."""

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no text encoder")
