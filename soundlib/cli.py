"""Soundlib CLI.

Examples:
    # Scan a folder, embed with Discogs-Effnet, write DuckDB.
    SOUNDLIB_DISCOGS_EMBED=./models/discogs-effnet-bs64-1.pb \\
    SOUNDLIB_DISCOGS_CLASSIFIER=./models/genre_discogs400-discogs-effnet-1.pb \\
        soundlib vectorize ~/Music --db library.duckdb

    # Quick sanity run with no model downloads.
    soundlib vectorize ./some_folder --db library.duckdb --backend librosa

    # UMAP + HDBSCAN clustering over stored embeddings.
    soundlib cluster --db library.duckdb --backend discogs-effnet

    # Find similar tracks.
    soundlib similar <file_id> --db library.duckdb --backend discogs-effnet

    # Text query (requires clap backend + embeddings).
    soundlib search "dark minimal techno" --db library.duckdb
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", required=True, help="path to DuckDB file")
    p.add_argument(
        "--backend", default="discogs-effnet",
        help="embedding backend (discogs-effnet, clap, mert, librosa)",
    )
    p.add_argument("-v", "--verbose", action="store_true")


def _log_level(ns: argparse.Namespace) -> int:
    return logging.DEBUG if ns.verbose else logging.INFO


def _cmd_vectorize(ns: argparse.Namespace) -> int:
    from soundlib.pipeline import vectorize_directory
    stats = vectorize_directory(
        root=ns.path,
        db_path=ns.db,
        backend_name=ns.backend,
        skip_existing=not ns.force,
        write_tags=ns.write_tags,
        top_k_genres_to_tag=ns.top_k,
    )
    print(
        f"seen={stats.seen} embedded={stats.embedded} "
        f"skipped={stats.skipped} failed={stats.failed}"
    )
    return 0


def _cmd_cluster(ns: argparse.Namespace) -> int:
    from soundlib.pipeline import cluster_library
    n = cluster_library(
        db_path=ns.db,
        backend_name=ns.backend,
        min_cluster_size=ns.min_cluster_size,
        umap_neighbors=ns.umap_neighbors,
    )
    print(f"found {n} cluster(s)")
    return 0


def _cmd_similar(ns: argparse.Namespace) -> int:
    from soundlib.db import SoundlibDB
    from soundlib.search import similar_to_track
    with SoundlibDB(ns.db) as db:
        rows = similar_to_track(db, ns.file_id, ns.backend, k=ns.k)
        _dump(rows, ns.json)
    return 0


def _cmd_search(ns: argparse.Namespace) -> int:
    from soundlib.db import SoundlibDB
    from soundlib.search import search_by_text
    with SoundlibDB(ns.db) as db:
        rows = search_by_text(db, ns.query, backend=ns.backend, k=ns.k)
        _dump(rows, ns.json)
    return 0


def _cmd_list_backends(_: argparse.Namespace) -> int:
    from soundlib.embeddings import list_backends
    for name in list_backends():
        print(name)
    return 0


def _dump(rows: list[tuple[str, float]], as_json: bool) -> None:
    if as_json:
        json.dump(
            [{"file_id": fid, "score": score} for fid, score in rows],
            sys.stdout, indent=2,
        )
        sys.stdout.write("\n")
    else:
        for fid, score in rows:
            print(f"{score:+.4f}\t{fid}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soundlib", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("vectorize", help="scan a directory and embed audio")
    pv.add_argument("path", help="directory (or single audio file) to scan")
    _add_common(pv)
    pv.add_argument("--force", action="store_true", help="re-embed tracks that already have a vector")
    pv.add_argument("--write-tags", action="store_true", help="write predicted genres back to file tags")
    pv.add_argument("--top-k", type=int, default=3, help="number of top genres to write as tags")
    pv.set_defaults(func=_cmd_vectorize)

    pc = sub.add_parser("cluster", help="UMAP + HDBSCAN over stored embeddings")
    _add_common(pc)
    pc.add_argument("--min-cluster-size", type=int, default=10)
    pc.add_argument("--umap-neighbors", type=int, default=15)
    pc.set_defaults(func=_cmd_cluster)

    ps = sub.add_parser("similar", help="nearest neighbors for a given track")
    ps.add_argument("file_id")
    _add_common(ps)
    ps.add_argument("-k", type=int, default=10)
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=_cmd_similar)

    pq = sub.add_parser("search", help="text → nearest neighbors (CLAP)")
    pq.add_argument("query")
    _add_common(pq)
    pq.add_argument("-k", type=int, default=10)
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(func=_cmd_search)

    pb = sub.add_parser("backends", help="list available embedding backends")
    pb.set_defaults(func=_cmd_list_backends)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    logging.basicConfig(level=_log_level(ns) if hasattr(ns, "verbose") else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
