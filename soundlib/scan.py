"""Directory walker that yields audio file paths."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from soundlib.audio import AUDIO_EXTENSIONS


@dataclass(frozen=True)
class Track:
    path: str
    size: int
    mtime: float
    file_id: str  # stable id = sha1(absolute_path)


def iter_audio_files(root: str | Path, follow_symlinks: bool = False) -> Iterator[Track]:
    root = Path(root).expanduser().resolve()
    if root.is_file():
        yield _make_track(root)
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames.sort()
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext not in AUDIO_EXTENSIONS:
                continue
            p = Path(dirpath) / name
            try:
                yield _make_track(p)
            except FileNotFoundError:
                continue


def _make_track(p: Path) -> Track:
    st = p.stat()
    abspath = str(p.resolve())
    file_id = hashlib.sha1(abspath.encode("utf-8")).hexdigest()
    return Track(path=abspath, size=st.st_size, mtime=st.st_mtime, file_id=file_id)
