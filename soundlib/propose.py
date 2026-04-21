"""Library-wide proposal: cluster the library, name each cluster from Discogs
hierarchical labels, and propose a target directory layout.

The proposal is a plain-text JSON file you can hand-edit before `apply`.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from soundlib.db import SoundlibDB


PROPOSAL_VERSION = 1


@dataclass
class ClusterSummary:
    cluster_id: int
    name: str                  # e.g. "Electronic/Minimal Techno"
    n_tracks: int
    top_genres: list[tuple[str, float]]  # aggregated probabilities
    confidence: float          # max top-genre prob share

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "top_genres": [(g, round(p, 4)) for g, p in self.top_genres],
            "confidence": round(self.confidence, 4),
        }


@dataclass
class MovePlan:
    file_id: str
    src: str
    dst: str
    cluster_id: int


@dataclass
class Proposal:
    root: str                  # target root where moves go
    backend: str
    version: int
    created_at: str
    clusters: list[ClusterSummary]
    moves: list[MovePlan]
    unassigned: list[str]      # file_ids with no cluster (HDBSCAN noise)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "root": self.root,
            "backend": self.backend,
            "clusters": [c.as_dict() for c in self.clusters],
            "moves": [asdict(m) for m in self.moves],
            "unassigned": list(self.unassigned),
        }


# -- building a proposal ------------------------------------------------------

def propose_layout(
    db: SoundlibDB,
    *,
    backend: str,
    target_root: str | Path,
    min_cluster_size: int = 8,
    umap_neighbors: int = 15,
    unclassified_dir: str = "Unclassified",
    max_depth: int = 2,
) -> Proposal:
    """Cluster embeddings via UMAP+HDBSCAN, name clusters, build move plan."""
    from soundlib.cluster import hdbscan_cluster, umap_project

    ids, mat = db.load_embeddings(backend)
    if len(ids) < min_cluster_size:
        return Proposal(
            root=str(target_root), backend=backend, version=PROPOSAL_VERSION,
            created_at=_now(), clusters=[], moves=[], unassigned=list(ids),
        )

    # Persist UMAP coords for the UI even if small library.
    coords = umap_project(mat, n_neighbors=min(umap_neighbors, max(2, len(ids) - 1)))
    labels = hdbscan_cluster(mat, min_cluster_size=min_cluster_size)
    db.write_clusters(backend, ids, labels, coords)

    paths = _load_paths(db, ids)
    genres = _load_genres(db, backend, ids)

    # Group file indices by cluster.
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        by_cluster[int(lab)].append(i)

    clusters: list[ClusterSummary] = []
    moves: list[MovePlan] = []
    unassigned: list[str] = []
    root = str(Path(target_root).expanduser())

    for cid, idxs in sorted(by_cluster.items()):
        members_ids = [ids[i] for i in idxs]
        if cid < 0:
            # HDBSCAN noise → unclassified folder (still proposed, not skipped).
            name = unclassified_dir
            clusters.append(ClusterSummary(
                cluster_id=cid, name=name, n_tracks=len(idxs),
                top_genres=[], confidence=0.0,
            ))
            for fid in members_ids:
                unassigned.append(fid)
                src = paths[fid]
                dst = _safe_dest(root, name, src)
                moves.append(MovePlan(file_id=fid, src=src, dst=dst, cluster_id=cid))
            continue

        agg, top1_labels = _aggregate_cluster_genres(members_ids, genres)
        name = _name_cluster(top1_labels, max_depth=max_depth) or f"Cluster_{cid:03d}"
        top_genres = sorted(agg.items(), key=lambda kv: -kv[1])[:5]
        total = sum(agg.values()) or 1.0
        confidence = (top_genres[0][1] / total) if top_genres else 0.0

        clusters.append(ClusterSummary(
            cluster_id=cid, name=name, n_tracks=len(idxs),
            top_genres=[(g, p / len(idxs)) for g, p in top_genres],
            confidence=float(confidence),
        ))
        for fid in members_ids:
            src = paths[fid]
            dst = _safe_dest(root, name, src)
            moves.append(MovePlan(file_id=fid, src=src, dst=dst, cluster_id=cid))

    return Proposal(
        root=root, backend=backend, version=PROPOSAL_VERSION, created_at=_now(),
        clusters=clusters, moves=moves, unassigned=unassigned,
    )


# -- serialize ----------------------------------------------------------------

def save_proposal(proposal: Proposal, path: str | Path) -> None:
    Path(path).write_text(json.dumps(proposal.as_dict(), indent=2), encoding="utf-8")


def load_proposal(path: str | Path) -> Proposal:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clusters = [ClusterSummary(**c) for c in data["clusters"]]
    moves = [MovePlan(**m) for m in data["moves"]]
    return Proposal(
        root=data["root"],
        backend=data["backend"],
        version=int(data.get("version", PROPOSAL_VERSION)),
        created_at=data.get("created_at", _now()),
        clusters=clusters,
        moves=moves,
        unassigned=list(data.get("unassigned", [])),
    )


# -- cluster naming -----------------------------------------------------------

_SPLIT_RE = re.compile(r"\s*(?:---|/|>|:)\s*")
_SAFE_RE = re.compile(r"[^\w\-. ()&']+", flags=re.UNICODE)


def _name_cluster(
    top1_labels: dict[str, float], max_depth: int = 2
) -> str:
    """Turn Discogs labels like 'Electronic---Minimal Techno' into
    'Electronic/Minimal Techno' and pick the dominant one."""
    if not top1_labels:
        return ""
    # Build hierarchical weight table. Each label contributes to every ancestor path.
    hier_weights: dict[tuple[str, ...], float] = defaultdict(float)
    for label, weight in top1_labels.items():
        parts = _split_label(label)[:max_depth]
        if not parts:
            continue
        for depth in range(1, len(parts) + 1):
            hier_weights[tuple(parts[:depth])] += weight
    # Prefer the deepest name that still accounts for ≥60% of the best-at-that-depth.
    best_path: tuple[str, ...] = ()
    for depth in range(1, max_depth + 1):
        at_depth = {k: v for k, v in hier_weights.items() if len(k) == depth}
        if not at_depth:
            break
        total_at_depth = sum(at_depth.values())
        top_path, top_weight = max(at_depth.items(), key=lambda kv: kv[1])
        if total_at_depth and top_weight / total_at_depth >= 0.6:
            best_path = top_path
        else:
            break
    if not best_path:
        # Fall back to the single heaviest label at depth 1.
        at_depth1 = {k: v for k, v in hier_weights.items() if len(k) == 1}
        if at_depth1:
            best_path = max(at_depth1.items(), key=lambda kv: kv[1])[0]
    return "/".join(_sanitize(p) for p in best_path)


def _split_label(label: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(label) if p.strip()]


def _sanitize(name: str) -> str:
    """Make a segment safe for a filesystem path."""
    out = _SAFE_RE.sub(" ", name).strip()
    out = re.sub(r"\s+", " ", out)
    return out or "Unknown"


def _safe_dest(root: str, folder: str, src: str) -> str:
    return str(Path(root) / folder / Path(src).name)


# -- aggregation helpers ------------------------------------------------------

def _aggregate_cluster_genres(
    file_ids: list[str], genres: dict[str, list[tuple[str, float]]]
) -> tuple[dict[str, float], dict[str, float]]:
    """Returns (full_agg_probs, top1_weight_by_label) for a cluster."""
    agg: dict[str, float] = defaultdict(float)
    top1: dict[str, float] = defaultdict(float)
    for fid in file_ids:
        glist = genres.get(fid, [])
        for label, prob in glist:
            agg[label] += prob
        if glist:
            top1[glist[0][0]] += glist[0][1]
    return agg, top1


def _load_paths(db: SoundlibDB, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    with db.cursor() as conn:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT file_id, path FROM tracks WHERE file_id IN ({placeholders})",
            ids,
        ).fetchall()
    return {fid: path for fid, path in rows}


def _load_genres(
    db: SoundlibDB, backend: str, ids: list[str]
) -> dict[str, list[tuple[str, float]]]:
    if not ids:
        return {}
    with db.cursor() as conn:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT file_id, label, prob
            FROM genres
            WHERE backend = ? AND file_id IN ({placeholders})
            ORDER BY prob DESC
            """,
            [backend, *ids],
        ).fetchall()
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for fid, label, prob in rows:
        out[fid].append((label, float(prob)))
    return out


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
