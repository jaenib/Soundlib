"""Live TUI for long-running phases.

One monitor wraps a whole pipeline run and shows:
    - phase header (Scan, Embed, Cluster, Analyze, ...)
    - progress bar (count, rate, ETA)
    - running stats panel: current file, top-5 detected genres so far, failures

Usage:
    with PipelineMonitor("vectorize ~/Music") as mon:
        mon.start_phase("scan", total=None)
        for track in iter_audio_files(...):
            mon.step(current=track.path)
        mon.finish_phase(found=N)

        mon.start_phase("embed", total=N)
        for track in tracks:
            ...
            mon.step(current=track.path, genre=top_genre)
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Iterable, Iterator, TypeVar

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

T = TypeVar("T")


@dataclass
class _Stats:
    phase: str = ""
    current: str = ""
    failures: int = 0
    genres: Counter = field(default_factory=Counter)
    custom: dict[str, str] = field(default_factory=dict)


class PipelineMonitor:
    def __init__(self, title: str, *, enabled: bool = True) -> None:
        self.title = title
        self.console = Console()
        self._enabled = enabled and self.console.is_terminal
        self._stats = _Stats()
        self._progress = Progress(
            TextColumn("[bold cyan]{task.description}", justify="left"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        self._task_id: int | None = None
        self._live: Live | None = None

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "PipelineMonitor":
        if self._enabled:
            self._live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *a) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(*a)
            self._live = None

    # -- phases -----------------------------------------------------------

    def start_phase(self, name: str, total: int | None = None) -> None:
        self._stats = _Stats(phase=name)
        if self._task_id is not None:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(name, total=total or 0)
        self._refresh()

    def step(
        self,
        n: int = 1,
        *,
        current: str | None = None,
        genre: str | None = None,
        failed: bool = False,
        **custom: str,
    ) -> None:
        if self._task_id is not None:
            self._progress.update(self._task_id, advance=n)
        if current is not None:
            self._stats.current = current
        if genre:
            self._stats.genres[genre] += 1
        if failed:
            self._stats.failures += 1
        if custom:
            self._stats.custom.update({k: str(v) for k, v in custom.items()})
        self._refresh()

    def finish_phase(self, **summary) -> None:
        if self._task_id is not None:
            if summary:
                self._stats.custom.update({k: str(v) for k, v in summary.items()})
            self._refresh()

    def log(self, msg: str) -> None:
        self.console.log(msg) if self._enabled else print(msg)

    # -- wrappers ---------------------------------------------------------

    def track(
        self, it: Iterable[T], total: int | None = None, phase: str = "working"
    ) -> Iterator[T]:
        self.start_phase(phase, total=total)
        for item in it:
            yield item
            self.step()

    # -- rendering --------------------------------------------------------

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self):
        stats = self._stats
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()

        if stats.current:
            table.add_row("current", _shorten(stats.current, 80))
        if stats.failures:
            table.add_row("failures", Text(str(stats.failures), style="bold red"))
        for k, v in stats.custom.items():
            table.add_row(k, str(v))
        if stats.genres:
            top = stats.genres.most_common(5)
            total = sum(stats.genres.values())
            bar_cells: list[str] = []
            for label, count in top:
                pct = count / total
                bar_cells.append(f"[green]{label}[/] [dim]{count} ({pct:.0%})[/]")
            table.add_row("top genres", "  ".join(bar_cells))

        return Group(
            Panel(Text(self.title, style="bold magenta"), border_style="magenta"),
            self._progress,
            Panel(table, title="live stats", border_style="cyan"),
        )


def _shorten(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    keep = n - 1
    return "…" + s[-keep:]


def noop_monitor() -> "nullcontext[PipelineMonitor]":
    """Returns a context manager with a dummy monitor that does nothing."""
    return nullcontext(_NullMonitor())  # type: ignore[return-value]


class _NullMonitor:
    def start_phase(self, *a, **kw): pass
    def step(self, *a, **kw): pass
    def finish_phase(self, *a, **kw): pass
    def log(self, msg: str) -> None: print(msg)
    def track(self, it, total=None, phase=""):
        for x in it:
            yield x
