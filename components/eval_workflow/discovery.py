"""Deterministic sampling and related-example discovery for trace review."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np


_TOKEN_RE = re.compile(r"[\w'-]{2,}", re.UNICODE)
_PREFERRED_CONTENT_FIELDS = (
    "input",
    "prompt",
    "question",
    "messages",
    "events",
    "trace",
    "trajectory",
    "output",
    "response",
    "completion",
    "final_answer",
    "answer",
    "reference",
    "expected",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def dataframe_to_records(dataframe: Any) -> list[dict[str, Any]]:
    """Convert a DataFrame-like object to stable, JSON-safe record dictionaries."""
    raw_records = dataframe.to_dict(orient="records")
    return [_json_safe(record) for record in raw_records]


def record_to_text(record: Mapping[str, Any]) -> str:
    """Extract review-relevant text while retaining enough structure for clustering."""
    values: list[str] = []
    used: set[str] = set()
    for field in _PREFERRED_CONTENT_FIELDS:
        if field not in record:
            continue
        used.add(field)
        value = record[field]
        if value in (None, ""):
            continue
        if isinstance(value, str):
            values.append(value)
        else:
            values.append(json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False))

    for field in sorted(record):
        if field in used:
            continue
        value = record[field]
        if value in (None, ""):
            continue
        if isinstance(value, (str, int, float, bool)):
            values.append(f"{field}: {value}")

    return "\n".join(values)


def _stable_bucket(token: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    return number % dimensions, 1.0 if number & 1 else -1.0


def _content_vector(text: str, dimensions: int = 96) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=float)
    tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
    counts = Counter(tokens)
    for token, count in counts.items():
        bucket, sign = _stable_bucket(token, dimensions)
        vector[bucket] += sign * (1.0 + math.log(count))
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _structural_vector(record: Mapping[str, Any], text: str) -> np.ndarray:
    nested_lists = sum(isinstance(value, list) for value in record.values())
    nested_dicts = sum(isinstance(value, Mapping) for value in record.values())
    return np.array(
        [
            math.log1p(len(text)),
            math.log1p(len(text.split())),
            math.log1p(text.count("\n") + 1),
            math.log1p(len(record)),
            math.log1p(nested_lists),
            math.log1p(nested_dicts),
            1.0 if any(field in record for field in ("events", "trace", "trajectory")) else 0.0,
            1.0 if any(field in record for field in ("reference", "expected", "expected_output")) else 0.0,
        ],
        dtype=float,
    )


def _feature_matrix(records: list[Mapping[str, Any]]) -> np.ndarray:
    if not records:
        return np.empty((0, 104), dtype=float)
    content = []
    structural = []
    for record in records:
        text = record_to_text(record)
        content.append(_content_vector(text))
        structural.append(_structural_vector(record, text))
    content_matrix = np.vstack(content)
    structural_matrix = np.vstack(structural)
    means = structural_matrix.mean(axis=0)
    stds = structural_matrix.std(axis=0)
    stds[stds == 0] = 1.0
    structural_matrix = (structural_matrix - means) / stds
    return np.hstack([content_matrix, structural_matrix])


def _kmeans(features: np.ndarray, cluster_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small NumPy-only KMeans with deterministic farthest-first initialization."""
    row_count = len(features)
    if row_count == 0:
        return np.array([], dtype=int), np.empty((0, features.shape[1]), dtype=float)
    cluster_count = max(1, min(cluster_count, row_count))
    if cluster_count == 1:
        return np.zeros(row_count, dtype=int), features.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(seed)
    centroid_indices = [int(rng.integers(0, row_count))]
    nearest_distance = np.sum((features - features[centroid_indices[0]]) ** 2, axis=1)
    while len(centroid_indices) < cluster_count:
        next_index = int(np.argmax(nearest_distance))
        if next_index in centroid_indices:
            remaining = [index for index in range(row_count) if index not in centroid_indices]
            next_index = remaining[0]
        centroid_indices.append(next_index)
        distance = np.sum((features - features[next_index]) ** 2, axis=1)
        nearest_distance = np.minimum(nearest_distance, distance)

    centroids = features[centroid_indices].copy()
    labels = np.zeros(row_count, dtype=int)
    for _ in range(50):
        distances = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for cluster_id in range(cluster_count):
            members = features[labels == cluster_id]
            if len(members):
                centroids[cluster_id] = members.mean(axis=0)
    return labels, centroids


def select_diverse_samples(
    records: Iterable[Mapping[str, Any]],
    sample_size: int = 20,
    seed: int = 42,
    cluster_fraction: float = 0.65,
    cluster_count: int | None = None,
    dimension_fields: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select cluster representatives plus random exploration records.

    The result is designed for failure-mode discovery. It must not be used to
    estimate production prevalence because cluster representatives are biased.
    """
    normalized_records = [dict(record) for record in records]
    row_count = len(normalized_records)
    if row_count == 0 or sample_size <= 0:
        return []
    sample_size = min(sample_size, row_count)
    if cluster_count is None:
        cluster_count = min(8, max(2, round(math.sqrt(row_count)))) if row_count > 1 else 1
    cluster_count = min(cluster_count, sample_size, row_count)

    features = _feature_matrix(normalized_records)
    labels, centroids = _kmeans(features, cluster_count, seed)
    representative_target = min(sample_size, max(cluster_count, round(sample_size * cluster_fraction)))

    ranked_by_cluster: dict[int, list[tuple[int, float]]] = {}
    for cluster_id in range(cluster_count):
        indices = np.flatnonzero(labels == cluster_id)
        ranked_by_cluster[cluster_id] = sorted(
            (
                (int(index), float(np.linalg.norm(features[index] - centroids[cluster_id])))
                for index in indices
            ),
            key=lambda item: (item[1], item[0]),
        )

    selections: list[dict[str, Any]] = []
    selected: set[int] = set()

    dimensions = [
        field
        for field in (dimension_fields or [])
        if any(field in record and record[field] not in (None, "") for record in normalized_records)
    ]
    if dimensions:
        dimension_groups: dict[tuple[str, ...], list[int]] = {}
        for index, record in enumerate(normalized_records):
            key = tuple(str(record.get(field, "(missing)")) for field in dimensions)
            dimension_groups.setdefault(key, []).append(index)
        # Rare combinations are represented first because they are easiest to miss.
        ranked_groups = sorted(dimension_groups.items(), key=lambda item: (len(item[1]), item[0]))
        dimension_target = min(len(ranked_groups), max(1, representative_target // 2))
        for values, indices in ranked_groups[:dimension_target]:
            group_features = features[indices]
            group_centroid = group_features.mean(axis=0)
            index = min(
                indices,
                key=lambda item: (float(np.linalg.norm(features[item] - group_centroid)), item),
            )
            selected.add(index)
            selections.append(
                {
                    "index": index,
                    "cluster": int(labels[index]),
                    "strategy": "dimension_coverage",
                    "reason": ", ".join(
                        f"{field}={value}" for field, value in zip(dimensions, values)
                    ),
                    "distance_to_centroid": round(
                        float(np.linalg.norm(features[index] - centroids[labels[index]])), 6
                    ),
                }
            )

    depth = 0
    while len(selections) < representative_target:
        added = False
        for cluster_id in range(cluster_count):
            candidates = ranked_by_cluster[cluster_id]
            if depth >= len(candidates):
                continue
            index, distance = candidates[depth]
            if index in selected:
                continue
            selected.add(index)
            selections.append(
                {
                    "index": index,
                    "cluster": cluster_id,
                    "strategy": "cluster_representative",
                    "reason": f"Representative of cluster {cluster_id + 1}",
                    "distance_to_centroid": round(distance, 6),
                }
            )
            added = True
            if len(selections) >= representative_target:
                break
        if not added:
            break
        depth += 1

    rng = np.random.default_rng(seed + 1)
    remaining = np.array([index for index in range(row_count) if index not in selected], dtype=int)
    if len(remaining):
        rng.shuffle(remaining)
    for index in remaining[: sample_size - len(selections)]:
        index = int(index)
        selections.append(
            {
                "index": index,
                "cluster": int(labels[index]),
                "strategy": "random_exploration",
                "reason": "Random exploration outside the representative set",
                "distance_to_centroid": round(
                    float(np.linalg.norm(features[index] - centroids[labels[index]])), 6
                ),
            }
        )
    return selections


def find_related_records(
    records: Iterable[Mapping[str, Any]],
    query_texts: Iterable[str],
    exclude_indices: Iterable[int] = (),
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return lexical-semantic candidates related to a failure-mode description/examples."""
    normalized_records = [dict(record) for record in records]
    queries = [text.strip() for text in query_texts if text and text.strip()]
    if not normalized_records or not queries or limit <= 0:
        return []

    query_vector = np.mean(np.vstack([_content_vector(text) for text in queries]), axis=0)
    query_norm = np.linalg.norm(query_vector)
    if query_norm:
        query_vector = query_vector / query_norm

    excluded = {int(index) for index in exclude_indices}
    candidates = []
    for index, record in enumerate(normalized_records):
        if index in excluded:
            continue
        score = float(np.dot(_content_vector(record_to_text(record)), query_vector))
        candidates.append(
            {
                "index": index,
                "score": round(max(0.0, score), 6),
                "reason": "Related to the accepted failure-mode description or examples",
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["index"]))
    return candidates[:limit]


def summarize_coverage(
    selections: Iterable[Mapping[str, Any]], reviewed_indices: Iterable[int]
) -> dict[str, Any]:
    """Summarize review coverage without claiming an unbiased failure rate."""
    selected = [dict(selection) for selection in selections]
    reviewed = {int(index) for index in reviewed_indices}
    clusters = sorted(
        {
            int(selection["cluster"])
            for selection in selected
            if int(selection.get("cluster", -1)) >= 0
        }
    )
    reviewed_clusters = sorted(
        {
            int(selection["cluster"])
            for selection in selected
            if int(selection.get("cluster", -1)) >= 0
            and int(selection["index"]) in reviewed
        }
    )
    return {
        "selected_count": len(selected),
        "reviewed_selected_count": sum(int(item["index"]) in reviewed for item in selected),
        "clusters_selected": len(clusters),
        "clusters_reviewed": len(reviewed_clusters),
        "unreviewed_clusters": [cluster for cluster in clusters if cluster not in reviewed_clusters],
        "prevalence_warning": (
            "Diverse discovery samples are intentionally biased and must not be used "
            "to estimate production failure rates."
        ),
    }
