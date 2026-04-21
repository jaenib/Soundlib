"""End-to-end orchestration.

Functions here stitch together scan → embed → DB, and are imported by the CLI.
They use PipelineMonitor to give a live TUI of what's happening."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from soundlib.audio import AudioLoadError, load_mono
from soundlib.db import SoundlibDB
from soundlib.embeddings import Backend, get_backend
from soundlib.progress import PipelineMonitor
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
    monitor: PipelineMonitor | None = None,
) -> VectorizeStats:
    mon = monitor or PipelineMonitor(f"vectorize {root}")
    stats = VectorizeStats()

    with mon:
        mon.start_phase("load backend", total=1)
        backend: Backend = get_backend(backend_name, **(backend_kwargs or {}))
        mon.step(current=f"backend={backend.name} sr={backend.sample_rate} dim={backend.dim}")
        mon.finish_phase()

        mon.start_phase("scan", total=None)
        tracks = []
        for t in iter_audio_files(root):
            tracks.append(t)
            mon.step(current=t.path)
        mon.finish_phase(found=len(tracks))

        mon.start_phase("embed", total=len(tracks))
        with SoundlibDB(db_path) as db:
            for track in tracks:
                stats.seen += 1
                if skip_existing and db.has_embedding(track.file_id, backend.name):
                    stats.skipped += 1
                    mon.step(current=track.path)
                    continue
                try:
                    audio = load_mono(track.path, backend.sample_rate)
                except AudioLoadError as e:
                    log.warning("decode failed: %s (%s)", track.path, e)
                    stats.failed += 1
                    mon.step(current=track.path, failed=True)
                    continue

                try:
                    result = backend.embed(audio)
                except Exception as e:  # noqa: BLE001
                    log.warning("embed failed: %s (%s)", track.path, e)
                    stats.failed += 1
                    mon.step(current=track.path, failed=True)
                    continue

                duration = len(audio) / backend.sample_rate
                db.upsert_track(
                    track.file_id, track.path, track.size, track.mtime,
                    duration=duration, sample_rate=backend.sample_rate,
                )
                db.upsert_embedding(track.file_id, backend.name, result.embedding)
                db.replace_genres(track.file_id, backend.name, result.genres)
                stats.embedded += 1

                top_genre = result.genres[0][0] if result.genres else None
                mon.step(current=track.path, genre=top_genre)

                if write_tags and result.genres:
                    from soundlib.tags import write_genres
                    labels = [g for g, _ in result.genres[:top_k_genres_to_tag]]
                    try:
                        write_genres(track.path, labels, mode="replace")
                    except Exception as e:  # noqa: BLE001
                        log.warning("tag write failed: %s (%s)", track.path, e)
        mon.finish_phase(
            embedded=stats.embedded, skipped=stats.skipped, failed=stats.failed,
        )

    return stats


def cluster_library(
    db_path: str | Path,
    backend_name: str = "discogs-effnet",
    min_cluster_size: int = 10,
    umap_neighbors: int = 15,
    monitor: PipelineMonitor | None = None,
) -> int:
    """Run UMAP + HDBSCAN on stored embeddings; persist to clusters table."""
    from soundlib.cluster import hdbscan_cluster, umap_project

    mon = monitor or PipelineMonitor(f"cluster {db_path}")
    with mon:
        with SoundlibDB(db_path) as db:
            mon.start_phase("load embeddings", total=1)
            ids, mat = db.load_embeddings(backend_name)
            mon.step(current=f"{len(ids)} vectors × {mat.shape[1] if mat.size else 0} dim")
            mon.finish_phase()

            if len(ids) < min_cluster_size:
                mon.log(f"not enough embeddings to cluster ({len(ids)})")
                db.write_clusters(backend_name, ids, None, None)
                return 0

            mon.start_phase("umap", total=1)
            coords = umap_project(mat, n_neighbors=umap_neighbors)
            mon.step(current="2D projection complete")
            mon.finish_phase()

            mon.start_phase("hdbscan", total=1)
            labels = hdbscan_cluster(mat, min_cluster_size=min_cluster_size)
            n_clusters = int((labels.max() + 1) if labels.size and labels.max() >= 0 else 0)
            mon.step(current=f"{n_clusters} clusters")
            mon.finish_phase(clusters=n_clusters)

            db.write_clusters(backend_name, ids, labels, coords)
            return n_clusters
