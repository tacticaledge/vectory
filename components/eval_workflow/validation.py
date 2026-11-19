"""Human-label calibration for binary evaluators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_PASS_LABELS = {"pass", "passed", "true", "1", "yes", "good", "positive"}
_FAIL_LABELS = {"fail", "failed", "false", "0", "no", "bad", "negative"}
_BOOTSTRAP_CELLS_PER_CHUNK = 2_000_000


def normalize_binary_label(value: Any) -> str:
    if isinstance(value, bool):
        return "Pass" if value else "Fail"
    if isinstance(value, (int, float)) and value in {0, 1}:
        return "Pass" if int(value) == 1 else "Fail"
    normalized = str(value).strip().casefold()
    if normalized in _PASS_LABELS:
        return "Pass"
    if normalized in _FAIL_LABELS:
        return "Fail"
    raise ValueError(f"Unsupported binary label: {value!r}")


def validate_evaluator_labels(
    human_labels: Sequence[Any], evaluator_labels: Sequence[Any]
) -> dict[str, Any]:
    """Calculate classifier-style alignment metrics against trusted human labels."""
    if len(human_labels) != len(evaluator_labels):
        raise ValueError("Human and evaluator label counts must match")
    if not human_labels:
        raise ValueError("At least one labeled example is required")
    human = np.array([normalize_binary_label(value) for value in human_labels])
    predicted = np.array([normalize_binary_label(value) for value in evaluator_labels])

    tp = int(np.sum((human == "Pass") & (predicted == "Pass")))
    fn = int(np.sum((human == "Pass") & (predicted == "Fail")))
    tn = int(np.sum((human == "Fail") & (predicted == "Fail")))
    fp = int(np.sum((human == "Fail") & (predicted == "Pass")))
    pass_total = tp + fn
    fail_total = tn + fp
    tpr = tp / pass_total if pass_total else None
    tnr = tn / fail_total if fail_total else None
    accuracy = (tp + tn) / len(human)
    balanced_accuracy = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    pass_precision = tp / (tp + fp) if tp + fp else None
    fail_precision = tn / (tn + fn) if tn + fn else None
    return {
        "examples": len(human),
        "confusion_matrix": {
            "true_pass": tp,
            "false_fail": fn,
            "true_fail": tn,
            "false_pass": fp,
        },
        "tpr": tpr,
        "tnr": tnr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class": {
            "Pass": {
                "precision": pass_precision,
                "recall": tpr,
                "f1": (
                    2 * pass_precision * tpr / (pass_precision + tpr)
                    if pass_precision is not None and tpr is not None and pass_precision + tpr
                    else None
                ),
            },
            "Fail": {
                "precision": fail_precision,
                "recall": tnr,
                "f1": (
                    2 * fail_precision * tnr / (fail_precision + tnr)
                    if fail_precision is not None and tnr is not None and fail_precision + tnr
                    else None
                ),
            },
        },
        "class_counts": {"Pass": pass_total, "Fail": fail_total},
    }


def bootstrap_metric_intervals(
    human_labels: Sequence[Any],
    evaluator_labels: Sequence[Any],
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, list[float] | None]:
    """Bootstrap 95% intervals for TPR and TNR."""
    if iterations <= 0:
        return {"tpr": None, "tnr": None}
    human = np.array([normalize_binary_label(value) for value in human_labels])
    predicted = np.array([normalize_binary_label(value) for value in evaluator_labels])
    if len(human) != len(predicted) or not len(human):
        raise ValueError("Equal, non-empty human and evaluator labels are required")

    rng = np.random.default_rng(seed)
    measurements: dict[str, list[float]] = defaultdict(list)
    pass_indices = np.flatnonzero(human == "Pass")
    fail_indices = np.flatnonzero(human == "Fail")
    for _ in range(iterations):
        if len(pass_indices):
            sampled = rng.choice(pass_indices, size=len(pass_indices), replace=True)
            measurements["tpr"].append(float(np.mean(predicted[sampled] == "Pass")))
        if len(fail_indices):
            sampled = rng.choice(fail_indices, size=len(fail_indices), replace=True)
            measurements["tnr"].append(float(np.mean(predicted[sampled] == "Fail")))
    return {
        metric: (
            [float(value) for value in np.percentile(values, [2.5, 97.5])]
            if values
            else None
        )
        for metric, values in (("tpr", measurements["tpr"]), ("tnr", measurements["tnr"]))
    }


def corrected_success_rate(observed_pass_rate: float, tpr: float, tnr: float) -> float:
    """Correct an observed judge pass rate for judge error (Rogan-Gladen estimator)."""
    for name, value in (("Observed pass rate", observed_pass_rate), ("TPR", tpr), ("TNR", tnr)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    denominator = tpr + tnr - 1
    if denominator <= 0:
        raise ValueError(
            "Judge is no better than chance (TPR + TNR <= 1), so its pass rate cannot be corrected"
        )
    return float(min(1.0, max(0.0, (observed_pass_rate + tnr - 1) / denominator)))


def estimate_success_rate(
    judge_labels: Sequence[Any],
    human_labels: Sequence[Any],
    evaluator_labels: Sequence[Any],
    *,
    iterations: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate the true pass rate of unlabeled outputs from imperfect judge labels.

    ``judge_labels`` are the evaluator's verdicts on the unlabeled population.
    ``human_labels`` and ``evaluator_labels`` are the held-out test pairs used to
    measure judge accuracy. The 95% interval bootstraps the test pairs and the
    judged population together, so it reflects uncertainty in TPR, TNR, and the
    observed pass rate.
    Resamples that lack a class or leave the judge at or below chance cannot be
    corrected; they count as 0 for the lower bound and 1 for the upper bound, so
    they widen the interval instead of being dropped.
    """
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    if not judge_labels:
        raise ValueError("At least one judge label is required")
    observed = np.array([normalize_binary_label(value) for value in judge_labels])
    observed_pass_rate = float(np.mean(observed == "Pass"))
    metrics = validate_evaluator_labels(human_labels, evaluator_labels)
    if metrics["tpr"] is None or metrics["tnr"] is None:
        raise ValueError("Test labels must include both Pass and Fail examples")
    estimate = corrected_success_rate(observed_pass_rate, metrics["tpr"], metrics["tnr"])

    human_pass = np.array([normalize_binary_label(value) == "Pass" for value in human_labels])
    judge_pass = np.array([normalize_binary_label(value) == "Pass" for value in evaluator_labels])
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    usable = np.empty(iterations, dtype=bool)
    # Resample in bounded chunks so memory stays flat for large test sets.
    chunk_size = max(1, _BOOTSTRAP_CELLS_PER_CHUNK // len(human_pass))
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        sampled_rate = rng.binomial(len(observed), observed_pass_rate, size=stop - start) / len(observed)
        indices = rng.integers(0, len(human_pass), size=(stop - start, len(human_pass)))
        sampled_human = human_pass[indices]
        sampled_judge = judge_pass[indices]
        pass_counts = sampled_human.sum(axis=1)
        fail_counts = len(human_pass) - pass_counts
        with np.errstate(divide="ignore", invalid="ignore"):
            tpr = (sampled_human & sampled_judge).sum(axis=1) / pass_counts
            tnr = (~sampled_human & ~sampled_judge).sum(axis=1) / fail_counts
            denominator = tpr + tnr - 1
            usable[start:stop] = (pass_counts > 0) & (fail_counts > 0) & (denominator > 0)
            estimates[start:stop] = np.clip((sampled_rate + tnr - 1) / denominator, 0.0, 1.0)
    lower = float(np.percentile(np.where(usable, estimates, 0.0), 2.5))
    upper = float(np.percentile(np.where(usable, estimates, 1.0), 97.5))

    return {
        "population_size": len(observed),
        "observed_pass_rate": observed_pass_rate,
        "corrected_pass_rate": estimate,
        "confidence_interval_95": [lower, upper],
        "judge_tpr": metrics["tpr"],
        "judge_tnr": metrics["tnr"],
        "test_examples": metrics["examples"],
        "bootstrap_iterations": iterations,
        "bootstrap_samples_uncorrectable": int(iterations - usable.sum()),
    }


def split_labeled_records(
    records: Iterable[Mapping[str, Any]],
    label_field: str,
    *,
    seed: int = 42,
    train_fraction: float = 0.15,
    dev_fraction: float = 0.45,
) -> dict[str, list[dict[str, Any]]]:
    """Create disjoint stratified train/dev/test splits for judge development."""
    if train_fraction <= 0 or dev_fraction <= 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("Train/dev fractions must be positive and leave room for a test split")
    grouped: dict[str, list[dict[str, Any]]] = {"Pass": [], "Fail": []}
    for record in records:
        item = dict(record)
        if label_field not in item:
            raise ValueError(f"Missing label field: {label_field}")
        grouped[normalize_binary_label(item[label_field])].append(item)
    if len(grouped["Pass"]) < 3 or len(grouped["Fail"]) < 3:
        raise ValueError("Stratified train/dev/test splits require at least three Pass and three Fail examples")

    rng = np.random.default_rng(seed)
    splits = {"train": [], "dev": [], "test": []}
    for label in ("Pass", "Fail"):
        items = grouped[label]
        order = rng.permutation(len(items))
        shuffled = [items[int(index)] for index in order]
        train_count = min(max(1, round(len(shuffled) * train_fraction)), len(shuffled) - 2)
        remaining = len(shuffled) - train_count
        dev_count = min(max(1, round(len(shuffled) * dev_fraction)), remaining - 1)
        train_end = train_count
        dev_end = train_count + dev_count
        splits["train"].extend(shuffled[:train_end])
        splits["dev"].extend(shuffled[train_end:dev_end])
        splits["test"].extend(shuffled[dev_end:])
    for split_name, items in splits.items():
        split_rng = np.random.default_rng(seed + {"train": 1, "dev": 2, "test": 3}[split_name])
        split_rng.shuffle(items)
    return splits
