"""End-to-end: scan a directory, embed, predict genres, store to DuckDB."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from soundlib.audio import AudioLoadError, load_mono
from soundlib.db import SoundlibDB
from soundlib.embeddings import Backend, get_backend
from soundlib.scan import iter_audio_files

log = logging.getLogger(__name__)


@dataclass
class VectorizeStats:
    seen: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0


def vectorize_directory(
    root: str | Path,
    db_path: str | Path,
    backend_name: str = "discogs-effnet",
    backend_kwargs: dict | None = None,
    skip_existing: bool = True,
    write_tags: bool = False,
    top_k_genres_to_tag: int = 3,
) -> VectorizeStats:
    backend: Backend = get_backend(backend_name, **(backend_kwargs or {}))
    stats = VectorizeStats()

    with SoundlibDB(db_path) as db:
        for track in tqdm(list(iter_audio_files(root)), desc=f"embed[{backend.name}]"):
            stats.seen += 1
            if skip_existing and db.has_embedding(track.file_id, backend.name):
                stats.skipped += 1
                continue
            try:
                audio = load_mono(track.path, backend.sample_rate)
            except AudioLoadError as e:
                log.warning("decode failed: %s (%s)", track.path, e)
                stats.failed += 1
                continue

            try:
                result = backend.embed(audio)
            except Exception as e:  # noqa: BLE001
                log.warning("embed failed: %s (%s)", track.path, e)
                stats.failed += 1
                continue

            duration = len(audio) / backend.sample_rate
            db.upsert_track(
                track.file_id, track.path, track.size, track.mtime,
                duration=duration, sample_rate=backend.sample_rate,
            )
            db.upsert_embedding(track.file_id, backend.name, result.embedding)
            db.replace_genres(track.file_id, backend.name, result.genres)
            stats.embedded += 1

            if write_tags and result.genres:
                from soundlib.tags import write_genres
                labels = [g for g, _ in result.genres[:top_k_genres_to_tag]]
                try:
                    write_genres(track.path, labels, mode="replace")
                except Exception as e:  # noqa: BLE001
                    log.warning("tag write failed: %s (%s)", track.path, e)

    return stats


def cluster_library(
    db_path: str | Path,
    backend_name: str = "discogs-effnet",
    min_cluster_size: int = 10,
    umap_neighbors: int = 15,
) -> int:
    """Run UMAP + HDBSCAN on stored embeddings; persist to clusters table.

    Returns the number of clusters found (excluding noise)."""
    from soundlib.cluster import hdbscan_cluster, umap_project

    with SoundlibDB(db_path) as db:
        ids, mat = db.load_embeddings(backend_name)
        if len(ids) < min_cluster_size:
            log.warning("not enough embeddings to cluster (%d)", len(ids))
            db.write_clusters(backend_name, ids, None, None)
            return 0
        coords = umap_project(mat, n_neighbors=umap_neighbors)
        labels = hdbscan_cluster(mat, min_cluster_size=min_cluster_size)
        db.write_clusters(backend_name, ids, labels, coords)
        n_clusters = int((labels.max() + 1) if labels.size and labels.max() >= 0 else 0)
        return n_clusters
