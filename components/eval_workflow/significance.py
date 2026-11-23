"""Paired significance tests and power planning for two runs on one dataset."""

from __future__ import annotations

from math import ceil, comb, sqrt
from statistics import NormalDist, fmean, variance
from typing import Any, Mapping, Sequence

import numpy as np

from . import validation
from .validation import normalize_binary_label

DEFAULT_BOOTSTRAP_ITERATIONS = 10000
# A paired comparison needs two items before any spread between them exists,
# so no plan or report may rest on fewer.
MINIMUM_PAIRED_ITEMS = 2


def exact_sign_test_p_value(items_favoring_a: int, items_favoring_b: int) -> float:
    """Two-sided exact paired sign test p-value from the two counts of winners.

    Items the two runs tie on carry no information about which run is better, so
    the test reads only the items with a non-zero difference. The counts must
    come from independent paired units, which under repeated sampling means
    items rather than rows. Under the null those items split evenly, giving
    ``p = 2 * P(X <= min(b, c))`` for ``X ~ Binomial(b + c, 0.5)``, clipped at
    1.0 because doubling a tail above one half overshoots. With a single binary
    run per item each item is one discordant pair, so this is exactly the
    two-sided exact McNemar test; with repeated runs the differences are
    fractional and only their signs are read, which is the sign test.
    """
    if items_favoring_a < 0 or items_favoring_b < 0:
        raise ValueError("Paired winner counts must not be negative")
    decided = items_favoring_a + items_favoring_b
    if decided == 0:
        return 1.0
    smaller = min(items_favoring_a, items_favoring_b)
    tail = sum(comb(decided, hits) for hits in range(smaller + 1)) / 2**decided
    return min(1.0, 2.0 * tail)


def paired_bootstrap_interval(
    item_differences: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 42,
    confidence: float = 0.95,
) -> list[float]:
    """Percentile bootstrap interval for the mean per-item difference.

    Items, not rows, are resampled: the repeated runs of one item move together,
    so repetition cannot pass for extra independent evidence. The same seed and
    item ordering always produce the same interval.
    """
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    if len(item_differences) < 2:
        raise ValueError("At least two paired items are required")
    differences = np.asarray(item_differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    # Resample in bounded chunks so memory stays flat for large item counts.
    chunk_size = max(1, validation._BOOTSTRAP_CELLS_PER_CHUNK // len(differences))
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1)
    remainder = (1.0 - confidence) / 2.0 * 100.0
    return [
        float(value) for value in np.percentile(means, [remainder, 100.0 - remainder])
    ]


def clustered_standard_error(item_differences: Sequence[float]) -> float:
    """Item-level cluster standard error, ``sqrt(Var(per-item mean difference) / n)``.

    This is the standard error to quote when an item is sampled several times:
    the repeated runs of one item are correlated, so only the item means count
    as independent observations.
    """
    if len(item_differences) < 2:
        raise ValueError("At least two paired items are required")
    return sqrt(variance(item_differences) / len(item_differences))


def naive_standard_error(
    paired_values: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> float:
    """Row-level standard error that treats every repeated run as independent.

    The repeats of one item are exchangeable, so no pairing of one run's repeats
    with the other's is privileged: the spread is averaged over every within-item
    pairing, which keeps the estimate independent of row order, and the row
    count rather than the larger pair count sets the sample size. With one row
    per run per item this is the ordinary paired standard error, so the design
    effect of unrepeated data is 1.
    """
    for values_a, values_b in paired_values:
        if len(values_a) != len(values_b):
            raise ValueError("Each item must carry the same number of rows per run")
    paired = [(values_a, values_b) for values_a, values_b in paired_values if values_a]
    rows = sum(len(values_a) for values_a, _ in paired)
    if rows < 2:
        raise ValueError("At least two paired rows are required")
    mean = (
        sum(
            len(values_a) * (fmean(values_a) - fmean(values_b))
            for values_a, values_b in paired
        )
        / rows
    )
    total = 0.0
    for values_a, values_b in paired:
        total += sum(
            (value_a - value_b - mean) ** 2
            for value_a in values_a
            for value_b in values_b
        ) / len(values_a)
    return sqrt(total / (rows - 1) / rows)


def _power_factor(alpha: float, power: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")
    normal = NormalDist()
    return normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)


def required_sample_size(
    effect: float,
    difference_sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Paired items needed to detect ``effect``, never fewer than two.

    ``n >= (z_{1-alpha/2} + z_{power})^2 * sigma_d^2 / delta^2``, so halving the
    effect worth detecting costs about four times as many items. A tiny or zero
    observed ``sigma_d`` can drive that bound below two, but a finite sample
    does not establish that the population differences have no spread, and a
    paired comparison needs two items before any spread can be estimated at
    all, so the recommendation is floored at the smallest usable plan.
    """
    if effect <= 0:
        raise ValueError("Target effect must be positive")
    if difference_sd < 0:
        raise ValueError("Difference standard deviation must not be negative")
    factor = _power_factor(alpha, power)
    return max(MINIMUM_PAIRED_ITEMS, ceil(factor**2 * difference_sd**2 / effect**2))


def minimum_detectable_effect(
    items: int,
    difference_sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest true difference ``items`` paired items can detect at this power."""
    if items < 1:
        raise ValueError("Item count must be at least 1")
    if difference_sd < 0:
        raise ValueError("Difference standard deviation must not be negative")
    return _power_factor(alpha, power) * difference_sd / sqrt(items)


def paired_run_outcomes(
    records: Sequence[Mapping[str, Any]],
    *,
    run_a: str,
    run_b: str,
    item_field: str = "item_id",
    run_field: str = "run_id",
    outcome_field: str = "judge_label",
) -> dict[str, Any]:
    """Pair two runs item by item in a repeated-sampling dataset.

    The item, not the row, is the paired unit: repeats collapse into the item's
    pass-rate difference, so results never depend on which repeat of one run a
    row happens to sit next to in the file. Items carrying rows from only one
    run are excluded instead of compared against nothing, and an item whose two
    runs have different repeat counts is refused, because the two pass rates
    would then rest on different amounts of evidence.

    Every row must name its run, but rows naming a third run are ignored and
    their other fields are never read.
    """
    if run_a == run_b:
        raise ValueError("Compared run ids must be different")
    grouped: dict[str, dict[str, list[float]]] = {}
    order: list[str] = []
    for index, record in enumerate(records):
        if run_field not in record:
            raise ValueError(f"Record {index} is missing columns: {run_field}")
        run = str(record[run_field])
        # Rows of other runs are skipped before their remaining fields are
        # read, so an unrelated run's malformed rows cannot abort a comparison
        # they take no part in.
        if run not in {run_a, run_b}:
            continue
        missing = [field for field in (item_field, outcome_field) if field not in record]
        if missing:
            raise ValueError(f"Record {index} is missing columns: {', '.join(missing)}")
        item = str(record[item_field])
        if item not in grouped:
            grouped[item] = {run_a: [], run_b: []}
            order.append(item)
        grouped[item][run].append(
            1.0 if normalize_binary_label(record[outcome_field]) == "Pass" else 0.0
        )

    items = []
    excluded = []
    for item in order:
        values_a = grouped[item][run_a]
        values_b = grouped[item][run_b]
        if not values_a or not values_b:
            excluded.append(item)
            continue
        if len(values_a) != len(values_b):
            raise ValueError(
                f"Item {item!r} has {len(values_a)} {run_a} rows and "
                f"{len(values_b)} {run_b} rows; paired comparison requires the "
                "same number of repeated runs per item"
            )
        pass_rate_a = fmean(values_a)
        pass_rate_b = fmean(values_b)
        items.append(
            {
                "item_id": item,
                "runs": len(values_a),
                "values_a": values_a,
                "values_b": values_b,
                "pass_rate_a": pass_rate_a,
                "pass_rate_b": pass_rate_b,
                "mean_difference": pass_rate_a - pass_rate_b,
            }
        )
    if not items:
        raise ValueError(f"No item carries rows from both {run_a} and {run_b}")
    item_differences = [item["mean_difference"] for item in items]
    return {
        "items": items,
        "item_differences": item_differences,
        "paired_values": [(item["values_a"], item["values_b"]) for item in items],
        "rows": sum(item["runs"] for item in items),
        "item_pass_rates_a": [item["pass_rate_a"] for item in items],
        "item_pass_rates_b": [item["pass_rate_b"] for item in items],
        # One signed observation per item, so correlated repeats of the same
        # item never count as separate evidence in the sign test.
        "items_favoring_a": sum(1 for difference in item_differences if difference > 0),
        "items_favoring_b": sum(1 for difference in item_differences if difference < 0),
        "excluded_unpaired_items": excluded,
    }


def compare_paired_runs(
    records: Sequence[Mapping[str, Any]],
    *,
    run_a: str,
    run_b: str,
    item_field: str = "item_id",
    run_field: str = "run_id",
    outcome_field: str = "judge_label",
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 42,
    alpha: float = 0.05,
    power: float = 0.8,
    target_effect: float | None = None,
) -> dict[str, Any]:
    """Compare two runs on the same items with a paired test, interval, and plan.

    Every statistic is item-level: the interval resamples items, the standard
    error clusters on items, and the sign test counts one signed observation
    per item. Repeated runs of one item therefore sharpen that item's estimate
    without passing for extra independent evidence.
    """
    outcomes = paired_run_outcomes(
        records,
        run_a=run_a,
        run_b=run_b,
        item_field=item_field,
        run_field=run_field,
        outcome_field=outcome_field,
    )
    item_differences = outcomes["item_differences"]
    if len(item_differences) < 2:
        raise ValueError("At least two paired items are required to compare two runs")
    clustered = clustered_standard_error(item_differences)
    naive = naive_standard_error(outcomes["paired_values"])
    difference_sd = sqrt(variance(item_differences))
    runs_per_item = [item["runs"] for item in outcomes["items"]]
    return {
        "items": len(item_differences),
        "paired_rows": outcomes["rows"],
        "runs_per_item_min": min(runs_per_item),
        "runs_per_item_max": max(runs_per_item),
        "excluded_unpaired_items": len(outcomes["excluded_unpaired_items"]),
        # Item-weighted like the difference below, so the reported delta is the
        # difference of the two pass rates even when repeat counts vary by item.
        "pass_rate_a": fmean(outcomes["item_pass_rates_a"]),
        "pass_rate_b": fmean(outcomes["item_pass_rates_b"]),
        "mean_difference": fmean(item_differences),
        "confidence_interval_95": paired_bootstrap_interval(
            item_differences, iterations=iterations, seed=seed
        ),
        "bootstrap_iterations": iterations,
        "seed": seed,
        "clustered_standard_error": clustered,
        "naive_standard_error": naive,
        # Above 1 the repeated runs are correlated, so row-level inference would
        # report a narrower interval than the data supports. A row-level error of
        # zero leaves the ratio undefined, which is reported as None rather than
        # as a design effect the data does not support.
        "design_effect": (clustered / naive) ** 2 if naive else None,
        "sign_test": {
            "paired_items": len(item_differences),
            "items_favoring_a": outcomes["items_favoring_a"],
            "items_favoring_b": outcomes["items_favoring_b"],
            "p_value": exact_sign_test_p_value(
                outcomes["items_favoring_a"], outcomes["items_favoring_b"]
            ),
        },
        "power": {
            "alpha": alpha,
            "power": power,
            "item_difference_sd": difference_sd,
            "minimum_detectable_effect": minimum_detectable_effect(
                len(item_differences), difference_sd, alpha=alpha, power=power
            ),
            "target_effect": target_effect,
            "required_items": (
                required_sample_size(target_effect, difference_sd, alpha=alpha, power=power)
                if target_effect is not None
                else None
            ),
        },
    }
