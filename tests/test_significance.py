import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from components.eval_workflow import (
    clustered_standard_error,
    compare_paired_runs,
    exact_mcnemar_p_value,
    minimum_detectable_effect,
    naive_standard_error,
    paired_bootstrap_interval,
    paired_run_outcomes,
    required_sample_size,
)


def paired_records(outcomes, run_a="candidate", run_b="baseline"):
    """Build run rows from {item: (candidate labels, baseline labels)} outcomes."""
    records = []
    for item, (values_a, values_b) in outcomes.items():
        for run, values in ((run_a, values_a), (run_b, values_b)):
            records.extend(
                {"item_id": item, "run_id": run, "judge_label": value} for value in values
            )
    return records


def test_exact_mcnemar_matches_hand_computed_binomial_tail():
    # b=10, c=2: 2 * (C(12,0) + C(12,1) + C(12,2)) / 2**12 = 158 / 4096.
    assert exact_mcnemar_p_value(10, 2) == pytest.approx(158 / 4096, abs=1e-9)
    assert exact_mcnemar_p_value(2, 10) == pytest.approx(158 / 4096, abs=1e-9)
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(3, 3) == 1.0
    with pytest.raises(ValueError):
        exact_mcnemar_p_value(-1, 2)


def test_paired_bootstrap_interval_repeats_exactly_for_one_seed():
    differences = [0.5, -0.25, 1.0, 0.0, 0.75, -0.5, 0.25, 0.5]

    first = paired_bootstrap_interval(differences, iterations=500, seed=7)
    second = paired_bootstrap_interval(differences, iterations=500, seed=7)

    assert first == second
    assert first != paired_bootstrap_interval(differences, iterations=500, seed=8)
    assert first[0] < sum(differences) / len(differences) < first[1]


def test_clustered_standard_error_exceeds_row_level_under_within_item_correlation():
    # Both repeated runs of an item agree, the strongest within-item correlation.
    correlated = {f"item-{index}": (["Pass", "Pass"], ["Fail", "Fail"]) for index in range(3)}
    correlated.update(
        {f"item-{index}": (["Fail", "Fail"], ["Pass", "Pass"]) for index in range(3, 6)}
    )
    outcomes = paired_run_outcomes(paired_records(correlated), run_a="candidate", run_b="baseline")

    clustered = clustered_standard_error(outcomes["item_differences"])
    row_level = naive_standard_error(outcomes["paired_values"])

    assert clustered > row_level
    single = paired_run_outcomes(
        paired_records({f"item-{index}": (["Pass"], ["Fail" if index % 3 else "Pass"]) for index in range(9)}),
        run_a="candidate",
        run_b="baseline",
    )
    assert clustered_standard_error(single["item_differences"]) == pytest.approx(
        naive_standard_error(single["paired_values"])
    )


def test_required_sample_size_falls_by_four_when_the_effect_doubles():
    small = required_sample_size(0.05, 0.2)
    doubled = required_sample_size(0.10, 0.2)

    assert small == 126
    assert abs(doubled - small / 4) <= 1
    assert minimum_detectable_effect(small, 0.2) == pytest.approx(0.05, abs=0.002)
    with pytest.raises(ValueError):
        required_sample_size(0.0, 0.2)
    with pytest.raises(ValueError):
        minimum_detectable_effect(64, 0.2, power=1.0)


def test_compare_paired_runs_reports_item_level_uncertainty():
    outcomes = {
        "item-1": (["Pass", "Pass"], ["Fail", "Fail"]),
        "item-2": (["Pass", "Pass"], ["Pass", "Pass"]),
        "item-3": (["Pass", "Pass"], ["Fail", "Pass"]),
        "item-4": (["Fail", "Fail"], ["Fail", "Fail"]),
        "item-5": (["Pass", "Fail"], ["Pass", "Pass"]),
        "item-6": (["Pass", "Pass"], ["Fail", "Fail"]),
    }
    records = paired_records(outcomes)
    records.append({"item_id": "item-7", "run_id": "candidate", "judge_label": "Pass"})

    report = compare_paired_runs(
        records, run_a="candidate", run_b="baseline", iterations=400, target_effect=0.1
    )

    assert report["items"] == 6
    assert report["paired_rows"] == 12
    assert report["excluded_unpaired_items"] == 1
    assert report["runs_per_item_min"] == report["runs_per_item_max"] == 2
    assert report["mean_difference"] == pytest.approx(2.0 / 6)
    assert report["pass_rate_a"] == pytest.approx(9 / 12)
    assert report["pass_rate_b"] == pytest.approx(5 / 12)
    assert report["mcnemar"]["paired_items"] == 6
    assert report["mcnemar"]["discordant_a_only"] == 3
    assert report["mcnemar"]["discordant_b_only"] == 1
    assert report["mcnemar"]["p_value"] == pytest.approx(exact_mcnemar_p_value(3, 1))
    assert report["clustered_standard_error"] > report["naive_standard_error"]
    assert report["design_effect"] > 1
    assert report["power"]["required_items"] == required_sample_size(
        0.1, report["power"]["item_difference_sd"]
    )
    assert report["power"]["minimum_detectable_effect"] > 0


def test_compare_paired_runs_ignores_row_order_within_an_item():
    # Both datasets hold the same outcomes per item; only the order of one
    # item's rows differs, which must not move any reported statistic.
    ordered = {
        "item-1": (["Pass", "Fail"], ["Pass", "Fail"]),
        "item-2": (["Pass", "Pass"], ["Pass", "Fail"]),
        "item-3": (["Pass", "Fail"], ["Pass", "Fail"]),
    }
    shuffled = dict(ordered)
    shuffled["item-1"] = (["Pass", "Fail"], ["Fail", "Pass"])
    shuffled["item-3"] = (["Fail", "Pass"], ["Pass", "Fail"])

    first = compare_paired_runs(
        paired_records(ordered), run_a="candidate", run_b="baseline", iterations=200
    )
    second = compare_paired_runs(
        paired_records(shuffled), run_a="candidate", run_b="baseline", iterations=200
    )

    assert first == second


def test_repeating_every_item_does_not_strengthen_the_mcnemar_evidence():
    outcomes = {
        "item-1": (["Pass"], ["Fail"]),
        "item-2": (["Pass"], ["Fail"]),
        "item-3": (["Pass"], ["Fail"]),
        "item-4": (["Fail"], ["Pass"]),
        "item-5": (["Pass"], ["Pass"]),
    }
    repeated = {
        item: (values_a * 4, values_b * 4) for item, (values_a, values_b) in outcomes.items()
    }

    once = compare_paired_runs(
        paired_records(outcomes), run_a="candidate", run_b="baseline", iterations=200
    )
    four_times = compare_paired_runs(
        paired_records(repeated), run_a="candidate", run_b="baseline", iterations=200
    )

    assert four_times["paired_rows"] == 4 * once["paired_rows"]
    assert four_times["mcnemar"] == once["mcnemar"]
    assert four_times["mcnemar"]["paired_items"] == 5
    assert four_times["mean_difference"] == once["mean_difference"]
    assert four_times["clustered_standard_error"] == pytest.approx(
        once["clustered_standard_error"]
    )
    # Row-level inference would have shrunk the standard error by sqrt(4).
    assert four_times["naive_standard_error"] == pytest.approx(
        once["naive_standard_error"] / 2, rel=0.2
    )
    assert four_times["design_effect"] > 3


def test_pass_rates_match_the_mean_difference_under_uneven_repeat_counts():
    outcomes = {
        "item-1": (["Pass", "Pass", "Pass", "Fail"], ["Fail", "Fail", "Fail", "Fail"]),
        "item-2": (["Pass"], ["Fail"]),
        "item-3": (["Fail"], ["Pass"]),
    }

    report = compare_paired_runs(
        paired_records(outcomes), run_a="candidate", run_b="baseline", iterations=200
    )

    assert report["runs_per_item_min"] == 1
    assert report["runs_per_item_max"] == 4
    assert report["pass_rate_a"] == pytest.approx((0.75 + 1.0 + 0.0) / 3)
    assert report["pass_rate_b"] == pytest.approx((0.0 + 0.0 + 1.0) / 3)
    assert report["pass_rate_a"] - report["pass_rate_b"] == pytest.approx(
        report["mean_difference"]
    )


def test_compare_paired_runs_refuses_unusable_pairings():
    with pytest.raises(ValueError, match="different"):
        compare_paired_runs(
            paired_records({"item-1": (["Pass"], ["Fail"])}),
            run_a="candidate",
            run_b="candidate",
        )
    with pytest.raises(ValueError, match="repeated runs per item"):
        compare_paired_runs(
            paired_records({"item-1": (["Pass", "Pass"], ["Fail"]), "item-2": (["Pass"], ["Fail"])}),
            run_a="candidate",
            run_b="baseline",
        )
    with pytest.raises(ValueError, match="both"):
        compare_paired_runs(
            [{"item_id": "item-1", "run_id": "candidate", "judge_label": "Pass"}],
            run_a="candidate",
            run_b="baseline",
        )
    with pytest.raises(ValueError, match="two paired items"):
        compare_paired_runs(
            paired_records({"item-1": (["Pass"], ["Fail"])}),
            run_a="candidate",
            run_b="baseline",
        )
    with pytest.raises(ValueError, match="missing columns"):
        compare_paired_runs(
            [{"item_id": "item-1", "run_id": "candidate"}],
            run_a="candidate",
            run_b="baseline",
        )


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
