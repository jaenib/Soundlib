"""Execute a Proposal.

Modes:
    dry-run  (default)  — print what would happen, no changes.
    symlink              — symlink src into dst. Non-destructive, reversible.
    copy                 — copy src to dst. Leaves the original alone.
    move                 — rename src → dst. Destructive.

Every non-dry-run action appends to a journal file (alongside the proposal)
so `soundlib rollback <journal>` can undo it.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from soundlib.propose import Proposal


APPLY_MODES = ("dry-run", "symlink", "copy", "move")


@dataclass
class ApplyReport:
    mode: str
    planned: int = 0
    executed: int = 0
    skipped: int = 0
    failed: int = 0
    conflicts: list[str] = field(default_factory=list)
    journal_path: str | None = None


def apply_proposal(
    proposal: Proposal,
    *,
    mode: str = "dry-run",
    journal_path: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> ApplyReport:
    if mode not in APPLY_MODES:
        raise ValueError(f"mode must be one of {APPLY_MODES}, got {mode!r}")

    report = ApplyReport(mode=mode)
    report.planned = len(proposal.moves)

    journal_fp = None
    if mode != "dry-run":
        jp = Path(journal_path) if journal_path else Path(f"soundlib-journal-{_ts()}.jsonl")
        report.journal_path = str(jp)
        journal_fp = jp.open("a", encoding="utf-8")
        _write_journal(journal_fp, {"header": {
            "created_at": _now(),
            "mode": mode,
            "proposal_root": proposal.root,
            "backend": proposal.backend,
        }})

    try:
        for i, move in enumerate(proposal.moves, start=1):
            src = Path(move.src)
            dst = Path(move.dst)

            if progress:
                progress(i, report.planned, str(src))

            if not src.exists():
                report.skipped += 1
                continue
            if dst.exists() and not overwrite:
                if dst.samefile(src):
                    report.skipped += 1
                    continue
                report.conflicts.append(str(dst))
                report.skipped += 1
                continue

            if mode == "dry-run":
                report.executed += 1
                continue

            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                op = _execute_one(src, dst, mode)
                report.executed += 1
                assert journal_fp is not None
                _write_journal(journal_fp, {"op": op})
            except Exception as e:  # noqa: BLE001
                report.failed += 1
                if journal_fp is not None:
                    _write_journal(journal_fp, {"error": str(e), "src": str(src), "dst": str(dst)})
    finally:
        if journal_fp is not None:
            journal_fp.close()

    return report


# -- rollback -----------------------------------------------------------------

def rollback(journal_path: str | Path) -> ApplyReport:
    """Undo a journal, in reverse order."""
    jp = Path(journal_path)
    entries: list[dict] = []
    with jp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    ops = [e["op"] for e in entries if "op" in e]
    report = ApplyReport(mode="rollback", planned=len(ops))

    for op in reversed(ops):
        try:
            _undo(op)
            report.executed += 1
        except Exception as e:  # noqa: BLE001
            report.failed += 1
            report.conflicts.append(f"{op}: {e}")

    return report


# -- internals ----------------------------------------------------------------

def _execute_one(src: Path, dst: Path, mode: str) -> dict:
    if mode == "symlink":
        os.symlink(src, dst)
        return {"op": "symlink", "src": str(src), "dst": str(dst)}
    if mode == "copy":
        shutil.copy2(src, dst)
        return {"op": "copy", "src": str(src), "dst": str(dst)}
    if mode == "move":
        shutil.move(str(src), str(dst))
        return {"op": "move", "src": str(src), "dst": str(dst)}
    raise ValueError(mode)


def _undo(op: dict) -> None:
    kind = op.get("op")
    src = Path(op["src"])
    dst = Path(op["dst"])
    if kind in ("symlink", "copy"):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    elif kind == "move":
        if dst.exists() and not src.exists():
            shutil.move(str(dst), str(src))
    else:
        raise ValueError(f"unknown journal op: {op}")


def _write_journal(fp, payload: dict) -> None:
    fp.write(json.dumps(payload) + "\n")
    fp.flush()


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
