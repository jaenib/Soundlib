"""Streamlit dashboard for a soundlib DuckDB.

Panels:
    1. UMAP scatter colored by cluster (or top genre), hover shows path + genres.
    2. Per-directory browser: cohesion, genre entropy, dominant genres, outliers.
    3. Proposal review: cluster-by-cluster diff of current vs proposed paths.

Launch via:
    soundlib ui --db library.duckdb
which internally runs:
    streamlit run soundlib/ui/streamlit_app.py -- --db library.duckdb
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


def _args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--db", required=True)
    p.add_argument("--proposal", default=None)
    known, _ = p.parse_known_args()
    return known


ARGS = _args()
DB_PATH = ARGS.db
PROPOSAL_PATH = ARGS.proposal


@st.cache_resource
def connect(path: str):
    return duckdb.connect(path, read_only=True)


@st.cache_data
def list_backends(_conn_key: str) -> list[str]:
    conn = connect(_conn_key)
    rows = conn.execute(
        "SELECT DISTINCT backend FROM embeddings ORDER BY backend"
    ).fetchall()
    return [r[0] for r in rows]


@st.cache_data
def scatter_frame(_conn_key: str, backend: str) -> pd.DataFrame:
    conn = connect(_conn_key)
    rows = conn.execute(
        """
        SELECT t.file_id, t.path, c.x, c.y, c.label AS cluster_label
        FROM tracks t
        JOIN clusters c ON c.file_id = t.file_id AND c.algo = ?
        WHERE c.x IS NOT NULL AND c.y IS NOT NULL
        """,
        [backend],
    ).fetchdf()

    # Attach top genre per file.
    if not rows.empty:
        ids = rows["file_id"].tolist()
        placeholders = ",".join("?" * len(ids))
        genres = conn.execute(
            f"""
            SELECT DISTINCT ON (file_id) file_id, label, prob
            FROM genres
            WHERE backend = ? AND file_id IN ({placeholders})
            ORDER BY file_id, prob DESC
            """,
            [backend, *ids],
        ).fetchdf()
        rows = rows.merge(genres, on="file_id", how="left")
        rows["top_genre"] = rows["label"].fillna("(none)")
        rows["cluster_name"] = rows["cluster_label"].apply(
            lambda x: "noise" if int(x) < 0 else f"C{int(x):02d}"
        )
        rows["filename"] = rows["path"].apply(lambda p: os.path.basename(p))
    return rows


@st.cache_data
def folder_frame(_conn_key: str, backend: str) -> pd.DataFrame:
    from soundlib.analyze import profile_directories
    from soundlib.db import SoundlibDB

    db = SoundlibDB(_conn_key)
    try:
        profiles = profile_directories(db, backend, min_tracks=2)
    finally:
        db.close()

    records = []
    for p in profiles:
        records.append({
            "directory": p.directory,
            "n_tracks": p.n_tracks,
            "cohesion": round(p.cohesion, 3),
            "genre_entropy": round(p.genre_entropy, 3),
            "top_genre": p.top_genres[0][0] if p.top_genres else "",
            "top_genre_prob": round(p.top_genres[0][1], 3) if p.top_genres else 0.0,
            "outliers": len(p.outliers),
        })
    return pd.DataFrame(records)


def main() -> None:
    st.set_page_config(page_title="soundlib", layout="wide")
    st.title("soundlib — music library vectorizer")
    st.caption(f"db: `{DB_PATH}`")

    backends = list_backends(DB_PATH)
    if not backends:
        st.warning("No embeddings yet. Run `soundlib vectorize` first.")
        return
    backend = st.sidebar.selectbox("Embedding backend", backends, index=0)
    view = st.sidebar.radio("View", ["UMAP scatter", "Folder browser", "Proposal review"])

    if view == "UMAP scatter":
        _render_scatter(backend)
    elif view == "Folder browser":
        _render_folders(backend)
    else:
        _render_proposal()


def _render_scatter(backend: str) -> None:
    df = scatter_frame(DB_PATH, backend)
    if df.empty:
        st.info("No UMAP coordinates for this backend yet. Run `soundlib cluster` to generate them.")
        return
    color_by = st.sidebar.radio("Color by", ["cluster_name", "top_genre"])
    fig = px.scatter(
        df, x="x", y="y", color=color_by,
        hover_data={"filename": True, "top_genre": True, "cluster_name": True,
                    "x": False, "y": False, "file_id": False},
        opacity=0.75, height=640,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_layout(legend=dict(orientation="v", y=1.0), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Genre distribution")
    counts = df["top_genre"].value_counts().head(20).reset_index()
    counts.columns = ["genre", "tracks"]
    st.bar_chart(counts, x="genre", y="tracks", height=300)


def _render_folders(backend: str) -> None:
    df = folder_frame(DB_PATH, backend)
    if df.empty:
        st.info("No folders to profile.")
        return
    st.caption("Cohesion = mean cosine(track, folder centroid). Entropy = normalized Shannon over top genres. Lower entropy = more focused folder.")
    st.dataframe(
        df.sort_values(["cohesion", "n_tracks"], ascending=[False, False]),
        use_container_width=True, height=640,
    )


def _render_proposal() -> None:
    path = PROPOSAL_PATH or st.text_input("Proposal JSON path", value="soundlib_proposal.json")
    if not path or not Path(path).exists():
        st.info("Generate a proposal with `soundlib propose --db DB --target /some/root --out proposal.json`.")
        return
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    st.markdown(f"**Proposal:** `{path}`  •  created `{data.get('created_at','?')}`  •  backend `{data.get('backend','?')}`")
    st.markdown(f"Target root: `{data['root']}`  •  {len(data['clusters'])} clusters  •  {len(data['moves'])} moves")

    cluster_rows = pd.DataFrame([
        {
            "id": c["cluster_id"],
            "name": c["name"],
            "tracks": c["n_tracks"],
            "confidence": c["confidence"],
            "top_genre": c["top_genres"][0][0] if c["top_genres"] else "",
        }
        for c in data["clusters"]
    ])
    st.subheader("Clusters")
    st.dataframe(cluster_rows, use_container_width=True)

    st.subheader("Moves (preview)")
    moves = pd.DataFrame(data["moves"])[:500]
    if not moves.empty:
        moves["filename"] = moves["src"].apply(lambda p: os.path.basename(p))
        moves["target_dir"] = moves["dst"].apply(lambda p: os.path.dirname(p))
        st.dataframe(moves[["filename", "target_dir", "cluster_id"]], use_container_width=True, height=480)


if __name__ == "__main__":
    main()
