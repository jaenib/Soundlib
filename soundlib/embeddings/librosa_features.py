"""Classical interpretable features via librosa. Low quality for genre but
useful as a zero-model-download fallback and for sanity-checking the pipeline."""
from __future__ import annotations

import numpy as np

from soundlib.embeddings.base import Backend, EmbeddingResult


LIBROSA_SR = 22050


class LibrosaFeatureBackend(Backend):
    name = "librosa"
    sample_rate = LIBROSA_SR
    dim = 13 * 2 + 12 * 2 + 7 * 2 + 2  # mfcc, chroma, contrast (mean+std) + tempo + zcr

    def __init__(self, n_mfcc: int = 13):
        try:
            import librosa  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "librosa required for the librosa backend. "
                "Install with: pip install soundlib[librosa]"
            ) from e
        self.n_mfcc = n_mfcc
        self.dim = n_mfcc * 2 + 12 * 2 + 7 * 2 + 2

    def embed(self, audio: np.ndarray) -> EmbeddingResult:
        import librosa
        y = np.asarray(audio, dtype=np.float32).ravel()
        sr = self.sample_rate

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])

        feats = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            contrast.mean(axis=1), contrast.std(axis=1),
            [zcr.mean(), zcr.std()],
        ]).astype(np.float32)

        return EmbeddingResult(
            embedding=feats,
            extra={"tempo_bpm": tempo},
        )
