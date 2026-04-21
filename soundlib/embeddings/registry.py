"""Registry maps a backend name to a lazy constructor so heavy deps only
import when that backend is actually requested."""
from __future__ import annotations

from typing import Callable

from soundlib.embeddings.base import Backend

_REGISTRY: dict[str, Callable[..., Backend]] = {}


def register(name: str):
    def deco(fn: Callable[..., Backend]) -> Callable[..., Backend]:
        _REGISTRY[name] = fn
        return fn
    return deco


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str, **kwargs) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backend {name!r}; available: {list_backends()}"
        )
    return _REGISTRY[name](**kwargs)


@register("discogs-effnet")
def _make_discogs(**kw) -> Backend:
    from soundlib.embeddings.essentia_discogs import DiscogsEffnetBackend
    return DiscogsEffnetBackend(**kw)


@register("clap")
def _make_clap(**kw) -> Backend:
    from soundlib.embeddings.clap import ClapBackend
    return ClapBackend(**kw)


@register("mert")
def _make_mert(**kw) -> Backend:
    from soundlib.embeddings.mert import MertBackend
    return MertBackend(**kw)


@register("librosa")
def _make_librosa(**kw) -> Backend:
    from soundlib.embeddings.librosa_features import LibrosaFeatureBackend
    return LibrosaFeatureBackend(**kw)
