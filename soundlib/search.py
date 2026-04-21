"""Nearest-neighbor search over stored embeddings."""
from __future__ import annotations

import numpy as np

from soundlib.db import SoundlibDB


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def cosine_topk(
    query: np.ndarray, matrix: np.ndarray, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (indices, similarities) sorted descending."""
    q = _normalize(np.atleast_2d(query.astype(np.float32)))
    m = _normalize(matrix.astype(np.float32))
    sims = (q @ m.T).ravel()
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def similar_to_track(
    db: SoundlibDB, file_id: str, backend: str, k: int = 10
) -> list[tuple[str, float]]:
    ids, mat = db.load_embeddings(backend)
    if not ids:
        return []
    try:
        anchor = ids.index(file_id)
    except ValueError:
        raise KeyError(f"no embedding for {file_id} under backend {backend!r}")
    idxs, sims = cosine_topk(mat[anchor], mat, k=k + 1)
    return [(ids[i], float(s)) for i, s in zip(idxs, sims) if ids[i] != file_id][:k]


def search_by_text(
    db: SoundlibDB, text: str, backend: str = "clap", k: int = 10
) -> list[tuple[str, float]]:
    from soundlib.embeddings import get_backend

    be = get_backend(backend)
    qvec = be.embed_text(text)
    ids, mat = db.load_embeddings(backend)
    if not ids:
        return []
    idxs, sims = cosine_topk(qvec, mat, k=k)
    return [(ids[i], float(s)) for i, s in zip(idxs, sims)]
