"""Read/write ID3/Vorbis/MP4 tags via mutagen."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def write_genres(
    path: str | Path,
    genres: Iterable[str],
    *,
    mode: str = "replace",
) -> None:
    """Write genre tags to a file.

    mode='replace' overwrites existing genre tags; mode='append' merges with any
    existing tags (deduped, preserving order).
    """
    from mutagen import File as MutagenFile

    genres = [g for g in genres if g]
    if not genres:
        return
    f = MutagenFile(str(path), easy=True)
    if f is None:
        raise ValueError(f"mutagen could not open {path}")

    existing = list(f.get("genre", [])) if mode == "append" else []
    combined: list[str] = []
    seen: set[str] = set()
    for g in list(existing) + list(genres):
        if g not in seen:
            seen.add(g)
            combined.append(g)
    f["genre"] = combined
    f.save()


def read_genres(path: str | Path) -> list[str]:
    from mutagen import File as MutagenFile

    f = MutagenFile(str(path), easy=True)
    if f is None:
        return []
    return list(f.get("genre", []))
