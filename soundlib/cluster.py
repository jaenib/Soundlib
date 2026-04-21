"""UMAP 2-D projection + HDBSCAN density clustering."""
from __future__ import annotations

import numpy as np


def umap_project(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise ImportError("pip install soundlib[cluster] for UMAP") from e
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X).astype(np.float32)


def hdbscan_cluster(
    X: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Returns label array; -1 means noise / unclustered."""
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError("pip install soundlib[cluster] for HDBSCAN") from e
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    return clusterer.fit_predict(X).astype(np.int32)
