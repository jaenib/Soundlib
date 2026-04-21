"""MERT (m-a-p/MERT-v1-95M or -330M) — music-specific transformer embeddings.
Mean-pooled hidden states over time. No genre head."""
from __future__ import annotations

import numpy as np

from soundlib.embeddings.base import Backend, EmbeddingResult


MERT_SR = 24000


class MertBackend(Backend):
    name = "mert"
    sample_rate = MERT_SR
    dim = 768  # 95M variant; 330M = 1024. Overwritten after load.

    def __init__(self, model_id: str = "m-a-p/MERT-v1-95M", use_cuda: bool | None = None):
        try:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except ImportError as e:
            raise ImportError(
                "transformers + torch required for the mert backend. "
                "Install with: pip install soundlib[mert]"
            ) from e

        self._torch = torch
        self._device = "cuda" if (use_cuda if use_cuda is not None else torch.cuda.is_available()) else "cpu"
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self._device).eval()
        self.sample_rate = int(self._processor.sampling_rate)
        self.dim = int(self._model.config.hidden_size)

    def embed(self, audio: np.ndarray) -> EmbeddingResult:
        torch = self._torch
        audio = np.asarray(audio, dtype=np.float32).ravel()
        inputs = self._processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(
                inputs["input_values"].to(self._device),
                output_hidden_states=False,
            )
        # Mean-pool final hidden states across time.
        vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        return EmbeddingResult(embedding=vec)
