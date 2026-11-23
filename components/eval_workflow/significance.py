"""Paired significance tests and power planning for two runs on one dataset."""

from __future__ import annotations

import random
from math import ceil, comb, floor, sqrt
from statistics import NormalDist, fmean, variance
from typing import Any, Mapping, Sequence

from .validation import normalize_binary_label

DEFAULT_BOOTSTRAP_ITERATIONS = 10000


def exact_mcnemar_p_value(discordant_a_only: int, discordant_b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant cells of a paired table.

    Concordant pairs carry no information about which run is better, so the test
    reads only the cells where the runs disagree. The counts must come from
    independent paired units, which under repeated sampling means items rather
    than rows. Under the null those pairs split evenly, giving
    ``p = 2 * P(X <= min(b, c))`` for ``X ~ Binomial(b + c, 0.5)``, clipped at
    1.0 because doubling a tail above one half overshoots.
    """
    if discordant_a_only < 0 or discordant_b_only < 0:
        raise ValueError("Discordant pair counts must not be negative")
    discordant = discordant_a_only + discordant_b_only
    if discordant == 0:
        return 1.0
    smaller = min(discordant_a_only, discordant_b_only)
    tail = sum(comb(discordant, hits) for hits in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


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
    rng = random.Random(seed)
    count = len(item_differences)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(count):
            total += item_differences[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    remainder = (1.0 - confidence) / 2.0 * 100.0
    return [_percentile(means, remainder), _percentile(means, 100.0 - remainder)]


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
    """Paired items needed to detect ``effect``.

    ``n >= (z_{1-alpha/2} + z_{power})^2 * sigma_d^2 / delta^2``, so halving the
    effect worth detecting costs about four times as many items.
    """
    if effect <= 0:
        raise ValueError("Target effect must be positive")
    if difference_sd < 0:
        raise ValueError("Difference standard deviation must not be negative")
    factor = _power_factor(alpha, power)
    return max(1, ceil(factor**2 * difference_sd**2 / effect**2))


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
    """
    if run_a == run_b:
        raise ValueError("Compared run ids must be different")
    grouped: dict[str, dict[str, list[float]]] = {}
    order: list[str] = []
    for index, record in enumerate(records):
        missing = [
            field for field in (item_field, run_field, outcome_field) if field not in record
        ]
        if missing:
            raise ValueError(f"Record {index} is missing columns: {', '.join(missing)}")
        run = str(record[run_field])
        if run not in {run_a, run_b}:
            continue
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
        # One discordant observation per item, so correlated repeats of the same
        # item never count as separate evidence in the McNemar test.
        "discordant_a_only": sum(1 for difference in item_differences if difference > 0),
        "discordant_b_only": sum(1 for difference in item_differences if difference < 0),
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
    error clusters on items, and the McNemar test counts one discordant
    observation per item. Repeated runs of one item therefore sharpen that
    item's estimate without passing for extra independent evidence.
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
        # report a narrower interval than the data supports.
        "design_effect": (clustered / naive) ** 2 if naive else 1.0,
        "mcnemar": {
            "paired_items": len(item_differences),
            "discordant_a_only": outcomes["discordant_a_only"],
            "discordant_b_only": outcomes["discordant_b_only"],
            "p_value": exact_mcnemar_p_value(
                outcomes["discordant_a_only"], outcomes["discordant_b_only"]
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
