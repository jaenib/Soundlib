"""LAION-CLAP — joint audio/text embedding. Lets you query 'dark minimal techno'
and get audio matches back. No genre head; use with a DB of known-good
prompts or plug into a classifier trained on the CLAP space."""
from __future__ import annotations

import numpy as np

from soundlib.embeddings.base import Backend, EmbeddingResult


CLAP_SR = 48000
CLAP_DIM = 512


class ClapBackend(Backend):
    name = "clap"
    sample_rate = CLAP_SR
    dim = CLAP_DIM

    def __init__(self, checkpoint: str | None = None, use_cuda: bool | None = None):
        try:
            import laion_clap  # type: ignore
            import torch
        except ImportError as e:
            raise ImportError(
                "laion-clap + torch required for the clap backend. "
                "Install with: pip install soundlib[clap]"
            ) from e

        device = "cuda" if (use_cuda if use_cuda is not None else torch.cuda.is_available()) else "cpu"
        self._torch = torch
        self._model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        # load_ckpt() with no argument pulls the default 630k-audioset checkpoint.
        if checkpoint:
            self._model.load_ckpt(checkpoint)
        else:
            self._model.load_ckpt()

    def embed(self, audio: np.ndarray) -> EmbeddingResult:
        audio = np.asarray(audio, dtype=np.float32).ravel()[None, :]
        vec = self._model.get_audio_embedding_from_data(x=audio, use_tensor=False)[0]
        return EmbeddingResult(embedding=vec.astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        vec = self._model.get_text_embedding([text], use_tensor=False)[0]
        return vec.astype(np.float32)
