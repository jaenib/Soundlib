"""Per-directory analysis: for each folder already containing tracks,
compute a profile (centroid, cohesion, dominant genres, outliers).

Use cases:
    - "Is this folder cohesive or a messy dump?"  → cohesion score
    - "What does this folder sound like?"         → top_genres
    - "Which tracks don't belong?"                → outliers
    - "Where should I put this new track?"        → recommend_folder
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from soundlib.db import SoundlibDB


@dataclass
class FolderProfile:
    directory: str
    n_tracks: int
    centroid: np.ndarray
    cohesion: float                     # mean cosine(track, centroid) ∈ [-1, 1]
    genre_entropy: float                # normalized Shannon entropy of top-1 labels
    top_genres: list[tuple[str, float]] # (label, avg prob over tracks)
    outliers: list[tuple[str, float]]   # (file_id, cosine to centroid) sorted asc
    members: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "directory": self.directory,
            "n_tracks": self.n_tracks,
            "cohesion": round(self.cohesion, 4),
            "genre_entropy": round(self.genre_entropy, 4),
            "top_genres": [(g, round(p, 4)) for g, p in self.top_genres],
            "outliers": [(fid, round(s, 4)) for fid, s in self.outliers],
        }


def profile_directories(
    db: SoundlibDB,
    backend: str,
    *,
    path_prefix: str | None = None,
    min_tracks: int = 3,
    top_n_outliers: int = 5,
    top_n_genres: int = 5,
) -> list[FolderProfile]:
    """Group tracks by parent directory, compute a profile per folder."""
    with db.cursor() as conn:
        where = "WHERE e.backend = ?"
        params: list = [backend]
        if path_prefix:
            where += " AND t.path LIKE ?"
            params.append(f"{path_prefix.rstrip(os.sep)}%")
        rows = conn.execute(
            f"""
            SELECT t.file_id, t.path, e.vector
            FROM tracks t
            JOIN embeddings e ON e.file_id = t.file_id
            {where}
            """,
            params,
        ).fetchall()

    by_dir: dict[str, list[tuple[str, str, list[float]]]] = defaultdict(list)
    for file_id, path, vector in rows:
        by_dir[os.path.dirname(path)].append((file_id, path, vector))

    # Pre-load genres for all file_ids in one query for speed.
    all_ids = [fid for members in by_dir.values() for (fid, _, _) in members]
    genre_map = _load_genres(db, backend, all_ids)

    profiles: list[FolderProfile] = []
    for directory, members in by_dir.items():
        if len(members) < min_tracks:
            continue
        ids = [m[0] for m in members]
        mat = np.asarray([m[2] for m in members], dtype=np.float32)
        centroid = mat.mean(axis=0)

        # Cohesion = mean cosine sim of each track to centroid.
        sims = _cosine_rows(mat, centroid)
        cohesion = float(sims.mean())

        # Outliers = lowest-similarity members.
        order = np.argsort(sims)
        outliers = [(ids[i], float(sims[i])) for i in order[:top_n_outliers]]

        # Top genres: aggregate prob across members, then rank.
        agg: dict[str, float] = defaultdict(float)
        for fid in ids:
            for label, prob in genre_map.get(fid, []):
                agg[label] += prob
        top_genres_full = sorted(agg.items(), key=lambda kv: -kv[1])
        top_genres = [(g, p / len(ids)) for g, p in top_genres_full[:top_n_genres]]

        # Genre entropy over top-1 label per track.
        top1_counts: dict[str, int] = defaultdict(int)
        for fid in ids:
            glist = genre_map.get(fid, [])
            if glist:
                top1_counts[glist[0][0]] += 1
        entropy = _normalized_entropy(list(top1_counts.values()))

        profiles.append(FolderProfile(
            directory=directory,
            n_tracks=len(ids),
            centroid=centroid,
            cohesion=cohesion,
            genre_entropy=entropy,
            top_genres=top_genres,
            outliers=outliers,
            members=ids,
        ))

    profiles.sort(key=lambda p: (-p.cohesion, -p.n_tracks))
    return profiles


def recommend_folder(
    db: SoundlibDB,
    backend: str,
    file_id: str,
    *,
    top_k: int = 3,
    min_tracks: int = 3,
) -> list[tuple[str, float]]:
    """Given an embedded track, return the top-k existing folders whose
    centroid is most similar. Useful when dropping a new file into a library."""
    with db.cursor() as conn:
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE file_id = ? AND backend = ?",
            [file_id, backend],
        ).fetchone()
    if row is None:
        raise KeyError(f"no embedding for {file_id} under backend {backend!r}")
    q = np.asarray(row[0], dtype=np.float32)

    profiles = profile_directories(db, backend, min_tracks=min_tracks)
    if not profiles:
        return []
    centroids = np.stack([p.centroid for p in profiles])
    sims = _cosine_rows(centroids, q)
    order = np.argsort(-sims)[:top_k]
    return [(profiles[i].directory, float(sims[i])) for i in order]


# -- internals ----------------------------------------------------------------

def _load_genres(
    db: SoundlibDB, backend: str, file_ids: list[str]
) -> dict[str, list[tuple[str, float]]]:
    if not file_ids:
        return {}
    with db.cursor() as conn:
        placeholders = ",".join("?" * len(file_ids))
        rows = conn.execute(
            f"""
            SELECT file_id, label, prob
            FROM genres
            WHERE backend = ? AND file_id IN ({placeholders})
            ORDER BY prob DESC
            """,
            [backend, *file_ids],
        ).fetchall()
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for fid, label, prob in rows:
        out[fid].append((label, float(prob)))
    return out


def _cosine_rows(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    m = mat.astype(np.float32)
    v = vec.astype(np.float32)
    m_norm = np.linalg.norm(m, axis=1)
    m_norm[m_norm == 0] = 1.0
    v_norm = np.linalg.norm(v) or 1.0
    return (m @ v) / (m_norm * v_norm)


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    h = -sum(p * math.log2(p) for p in probs)
    return h / math.log2(len(probs)) if len(probs) > 1 else 0.0
