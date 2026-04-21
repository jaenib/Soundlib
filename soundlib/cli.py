"""Soundlib CLI.

Workflows:

    1. Embed a library.
        SOUNDLIB_DISCOGS_EMBED=./models/discogs-effnet-bs64-1.pb \\
        SOUNDLIB_DISCOGS_CLASSIFIER=./models/genre_discogs400-discogs-effnet-1.pb \\
            soundlib vectorize ~/Music --db library.duckdb

        # Or use the zero-download fallback:
        soundlib vectorize ~/Music --db library.duckdb --backend librosa

    2a. Scan & propose a genre-folder layout (library-wide).
        soundlib propose --db library.duckdb --target ~/Sorted --out proposal.json
        soundlib apply proposal.json --mode symlink     # non-destructive
        soundlib apply proposal.json --mode move        # destructive, with journal

    2b. Per-directory analysis of what you already have.
        soundlib analyze --db library.duckdb --path ~/Music/Techno

    3. Browse visually.
        soundlib ui --db library.duckdb --proposal proposal.json

    4. Other:
        soundlib similar <file_id> --db library.duckdb
        soundlib search "dark minimal techno" --db library.duckdb
        soundlib rollback soundlib-journal-XXXX.jsonl
        soundlib backends
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser, require_backend: bool = True) -> None:
    p.add_argument("--db", required=True, help="path to DuckDB file")
    if require_backend:
        p.add_argument(
            "--backend", default="discogs-effnet",
            help="embedding backend (discogs-effnet, clap, mert, librosa)",
        )
    p.add_argument("-v", "--verbose", action="store_true")


def _log_level(ns: argparse.Namespace) -> int:
    return logging.DEBUG if getattr(ns, "verbose", False) else logging.INFO


# -- vectorize ----------------------------------------------------------------

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


# -- cluster ------------------------------------------------------------------

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


# -- similar / search ---------------------------------------------------------

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


# -- analyze (per-dir) --------------------------------------------------------

def _cmd_analyze(ns: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from soundlib.analyze import profile_directories
    from soundlib.db import SoundlibDB

    with SoundlibDB(ns.db) as db:
        profiles = profile_directories(
            db, ns.backend,
            path_prefix=ns.path,
            min_tracks=ns.min_tracks,
            top_n_outliers=ns.outliers,
            top_n_genres=ns.top_genres,
        )

    if ns.json:
        json.dump([p.as_dict() for p in profiles], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    console = Console()
    if not profiles:
        console.print("[yellow]no folders met the minimum track threshold[/]")
        return 0

    table = Table(title=f"per-directory analysis ({ns.backend})", expand=True)
    table.add_column("directory", overflow="fold")
    table.add_column("tracks", justify="right")
    table.add_column("cohesion", justify="right")
    table.add_column("entropy", justify="right")
    table.add_column("dominant genre(s)")
    table.add_column("outliers", justify="right")
    for p in profiles:
        genre_cell = ", ".join(f"{g} ({prob:.2f})" for g, prob in p.top_genres[:3])
        table.add_row(
            p.directory,
            str(p.n_tracks),
            f"{p.cohesion:+.3f}",
            f"{p.genre_entropy:.2f}",
            genre_cell or "—",
            str(len(p.outliers)),
        )
    console.print(table)
    return 0


# -- propose / apply / rollback ----------------------------------------------

def _cmd_propose(ns: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from soundlib.db import SoundlibDB
    from soundlib.propose import propose_layout, save_proposal

    with SoundlibDB(ns.db) as db:
        proposal = propose_layout(
            db,
            backend=ns.backend,
            target_root=ns.target,
            min_cluster_size=ns.min_cluster_size,
            umap_neighbors=ns.umap_neighbors,
            unclassified_dir=ns.unclassified_dir,
            max_depth=ns.max_depth,
        )

    out = Path(ns.out)
    save_proposal(proposal, out)

    console = Console()
    console.print(
        f"[green]wrote proposal[/]: [bold]{out}[/] — "
        f"{len(proposal.clusters)} clusters, {len(proposal.moves)} moves, "
        f"{len(proposal.unassigned)} unassigned"
    )
    table = Table(title="cluster summary", expand=True)
    table.add_column("id", justify="right")
    table.add_column("proposed name", overflow="fold")
    table.add_column("tracks", justify="right")
    table.add_column("confidence", justify="right")
    table.add_column("top genre")
    for c in proposal.clusters:
        top = c.top_genres[0][0] if c.top_genres else ""
        table.add_row(
            str(c.cluster_id), c.name, str(c.n_tracks),
            f"{c.confidence:.2f}", top,
        )
    console.print(table)
    console.print(
        f"[dim]next:[/] [bold]soundlib apply {out} --mode symlink[/] "
        f"(dry-run by default)"
    )
    return 0


def _cmd_apply(ns: argparse.Namespace) -> int:
    from rich.console import Console

    from soundlib.apply import apply_proposal
    from soundlib.propose import load_proposal

    proposal = load_proposal(ns.proposal)
    journal = ns.journal or f"{Path(ns.proposal).stem}-journal.jsonl"

    console = Console()
    last = {"i": 0}

    def progress_cb(i: int, total: int, current: str) -> None:
        if i - last["i"] >= max(1, total // 50):
            console.log(f"[{i}/{total}] {current}")
            last["i"] = i

    report = apply_proposal(
        proposal, mode=ns.mode, journal_path=journal, overwrite=ns.overwrite,
        progress=progress_cb,
    )
    console.print(
        f"mode=[bold]{report.mode}[/] planned={report.planned} "
        f"executed={report.executed} skipped={report.skipped} failed={report.failed}"
    )
    if report.conflicts:
        console.print(f"[yellow]conflicts:[/] {len(report.conflicts)} (first 5 shown)")
        for c in report.conflicts[:5]:
            console.print(f"  {c}")
    if report.journal_path:
        console.print(f"journal: [bold]{report.journal_path}[/]")
        console.print(f"[dim]undo:[/] soundlib rollback {report.journal_path}")
    return 0


def _cmd_rollback(ns: argparse.Namespace) -> int:
    from rich.console import Console
    from soundlib.apply import rollback
    report = rollback(ns.journal)
    Console().print(
        f"rollback planned={report.planned} executed={report.executed} "
        f"failed={report.failed}"
    )
    return 0


# -- ui -----------------------------------------------------------------------

def _cmd_ui(ns: argparse.Namespace) -> int:
    import shutil
    import subprocess

    app = Path(__file__).with_name("ui") / "streamlit_app.py"
    if not app.exists():
        print(f"streamlit app not found at {app}", file=sys.stderr)
        return 1
    if shutil.which("streamlit") is None:
        print("streamlit not installed. pip install soundlib[ui]", file=sys.stderr)
        return 1
    cmd = ["streamlit", "run", str(app), "--", "--db", ns.db]
    if ns.proposal:
        cmd += ["--proposal", ns.proposal]
    return subprocess.call(cmd)


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
    pv.add_argument("--force", action="store_true",
                    help="re-embed tracks that already have a vector")
    pv.add_argument("--write-tags", action="store_true",
                    help="write predicted genres back to file tags")
    pv.add_argument("--top-k", type=int, default=3,
                    help="number of top genres to write as tags")
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

    pa = sub.add_parser("analyze", help="per-directory cohesion + genre profile")
    _add_common(pa)
    pa.add_argument("--path", default=None,
                    help="restrict to folders under this path prefix")
    pa.add_argument("--min-tracks", type=int, default=3)
    pa.add_argument("--outliers", type=int, default=5,
                    help="top-N least-similar tracks to list per folder")
    pa.add_argument("--top-genres", type=int, default=5)
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=_cmd_analyze)

    pp = sub.add_parser("propose", help="cluster library and propose genre folders")
    _add_common(pp)
    pp.add_argument("--target", required=True,
                    help="root directory where proposed folders should live")
    pp.add_argument("--out", default="soundlib_proposal.json",
                    help="where to write the proposal JSON")
    pp.add_argument("--min-cluster-size", type=int, default=8)
    pp.add_argument("--umap-neighbors", type=int, default=15)
    pp.add_argument("--unclassified-dir", default="Unclassified")
    pp.add_argument("--max-depth", type=int, default=2,
                    help="max nesting depth for proposed folder names")
    pp.set_defaults(func=_cmd_propose)

    pap = sub.add_parser("apply", help="execute a proposal (dry-run by default)")
    pap.add_argument("proposal", help="path to a proposal JSON")
    pap.add_argument("--mode", default="dry-run",
                     choices=("dry-run", "symlink", "copy", "move"))
    pap.add_argument("--journal", default=None,
                     help="where to write the rollback journal (non-dry-run only)")
    pap.add_argument("--overwrite", action="store_true",
                     help="overwrite existing files at destination")
    pap.add_argument("-v", "--verbose", action="store_true")
    pap.set_defaults(func=_cmd_apply)

    pr = sub.add_parser("rollback", help="undo a previously applied proposal")
    pr.add_argument("journal", help="journal file written by `apply`")
    pr.add_argument("-v", "--verbose", action="store_true")
    pr.set_defaults(func=_cmd_rollback)

    pui = sub.add_parser("ui", help="open the Streamlit dashboard")
    pui.add_argument("--db", required=True)
    pui.add_argument("--proposal", default=None,
                     help="optional path to a proposal JSON to preview")
    pui.add_argument("-v", "--verbose", action="store_true")
    pui.set_defaults(func=_cmd_ui)

    pb = sub.add_parser("backends", help="list available embedding backends")
    pb.set_defaults(func=_cmd_list_backends)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    logging.basicConfig(level=_log_level(ns),
                        format="%(levelname)s %(name)s: %(message)s")
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
