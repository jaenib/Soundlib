"""DuckDB storage for tracks, embeddings, and predicted genre tags.

Schema:
    tracks(file_id PK, path, size, mtime, duration, sample_rate, scanned_at)
    embeddings(file_id, backend, dim, vector FLOAT[], PRIMARY KEY (file_id, backend))
    genres(file_id, backend, label, prob)
    clusters(file_id, algo, label, x, y)            -- UMAP coords + HDBSCAN label
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import duckdb
import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    file_id     VARCHAR PRIMARY KEY,
    path        VARCHAR NOT NULL,
    size        BIGINT,
    mtime       DOUBLE,
    duration    DOUBLE,
    sample_rate INTEGER,
    scanned_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    file_id VARCHAR NOT NULL,
    backend VARCHAR NOT NULL,
    dim     INTEGER NOT NULL,
    vector  FLOAT[] NOT NULL,
    PRIMARY KEY (file_id, backend)
);

CREATE TABLE IF NOT EXISTS genres (
    file_id VARCHAR NOT NULL,
    backend VARCHAR NOT NULL,
    label   VARCHAR NOT NULL,
    prob    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_genres_file    ON genres(file_id);
CREATE INDEX IF NOT EXISTS idx_genres_backend ON genres(backend);

CREATE TABLE IF NOT EXISTS clusters (
    file_id VARCHAR NOT NULL,
    algo    VARCHAR NOT NULL,
    label   INTEGER,
    x       REAL,
    y       REAL,
    PRIMARY KEY (file_id, algo)
);
"""


class SoundlibDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = duckdb.connect(self.path)
        self._conn.execute("PRAGMA enable_progress_bar=false")
        for stmt in filter(str.strip, SCHEMA.split(";")):
            self._conn.execute(stmt)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SoundlibDB":
        return self

    def __exit__(self, *a) -> None:
        self.close()

    # -- tracks -----------------------------------------------------------

    def upsert_track(
        self,
        file_id: str,
        path: str,
        size: int,
        mtime: float,
        duration: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO tracks (file_id, path, size, mtime, duration, sample_rate)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (file_id) DO UPDATE SET
                path = excluded.path,
                size = excluded.size,
                mtime = excluded.mtime,
                duration = COALESCE(excluded.duration, tracks.duration),
                sample_rate = COALESCE(excluded.sample_rate, tracks.sample_rate),
                scanned_at = now()
            """,
            [file_id, path, size, mtime, duration, sample_rate],
        )

    def has_embedding(self, file_id: str, backend: str) -> bool:
        (row,) = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE file_id = ? AND backend = ?",
            [file_id, backend],
        ).fetchone()
        return bool(row)

    # -- embeddings -------------------------------------------------------

    def upsert_embedding(
        self, file_id: str, backend: str, vector: np.ndarray
    ) -> None:
        vec = np.asarray(vector, dtype=np.float32).ravel().tolist()
        self._conn.execute(
            """
            INSERT INTO embeddings (file_id, backend, dim, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (file_id, backend) DO UPDATE SET
                dim = excluded.dim,
                vector = excluded.vector
            """,
            [file_id, backend, len(vec), vec],
        )

    def load_embeddings(
        self, backend: str
    ) -> tuple[list[str], np.ndarray]:
        rows = self._conn.execute(
            """
            SELECT e.file_id, e.vector
            FROM embeddings e
            WHERE e.backend = ?
            ORDER BY e.file_id
            """,
            [backend],
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r[0] for r in rows]
        mat = np.asarray([r[1] for r in rows], dtype=np.float32)
        return ids, mat

    # -- genres -----------------------------------------------------------

    def replace_genres(
        self,
        file_id: str,
        backend: str,
        genres: Iterable[tuple[str, float]],
    ) -> None:
        self._conn.execute(
            "DELETE FROM genres WHERE file_id = ? AND backend = ?",
            [file_id, backend],
        )
        rows = [(file_id, backend, label, float(prob)) for label, prob in genres]
        if rows:
            self._conn.executemany(
                "INSERT INTO genres (file_id, backend, label, prob) VALUES (?, ?, ?, ?)",
                rows,
            )

    def top_genres(
        self, file_id: str, backend: str, k: int = 5
    ) -> list[tuple[str, float]]:
        return self._conn.execute(
            """
            SELECT label, prob
            FROM genres
            WHERE file_id = ? AND backend = ?
            ORDER BY prob DESC
            LIMIT ?
            """,
            [file_id, backend, k],
        ).fetchall()

    # -- clusters ---------------------------------------------------------

    def write_clusters(
        self,
        algo: str,
        file_ids: list[str],
        labels: np.ndarray | None,
        coords: np.ndarray | None,
    ) -> None:
        self._conn.execute("DELETE FROM clusters WHERE algo = ?", [algo])
        rows = []
        for i, fid in enumerate(file_ids):
            lab = int(labels[i]) if labels is not None else None
            x = float(coords[i, 0]) if coords is not None else None
            y = float(coords[i, 1]) if coords is not None else None
            rows.append((fid, algo, lab, x, y))
        if rows:
            self._conn.executemany(
                "INSERT INTO clusters (file_id, algo, label, x, y) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    # -- iteration --------------------------------------------------------

    @contextmanager
    def cursor(self) -> Iterator[duckdb.DuckDBPyConnection]:
        yield self._conn

    def track_count(self) -> int:
        (n,) = self._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()
        return int(n)
