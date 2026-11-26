import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from components.evaluators.rubric import (
    RubricReport,
    aggregate_rubric_results,
    parse_rubric_response,
    parse_rubric_spec,
    render_rubric_checklist,
)


RUBRIC = {
    "schema_version": "1.0",
    "rubric_id": "refund-reply",
    "criteria": [
        {"id": "states_window", "text": "States the refund window.", "weight": 3},
        {"id": "cites_policy", "text": "Cites the policy section.", "weight": 2},
        {"id": "next_step", "text": "Gives the next concrete step.", "weight": 1},
        {
            "id": "invents_terms",
            "text": "Invents a refund term.",
            "weight": 2,
            "polarity": "negative",
        },
    ],
}


def recorded_response(critique="Specific, grounded critique.", **verdicts):
    """Build one recorded judge response the way a checklist judge returns it."""
    return json.dumps(
        {
            "critique": critique,
            "criteria": [
                {"id": identifier, "met": met} for identifier, met in verdicts.items()
            ],
        }
    )


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            {
                "rubric_id": "refund-reply",
                "criteria": [
                    {"id": "same", "text": "First.", "weight": 1},
                    {"id": "same", "text": "Second.", "weight": 2},
                ],
            },
            "ids must be unique",
        ),
        (
            {
                "rubric_id": "refund-reply",
                "criteria": [{"id": "states_window", "text": "States it.", "weight": "3"}],
            },
            "weight must be a number",
        ),
        (
            {
                "rubric_id": "refund-reply",
                "criteria": [{"id": "states_window", "text": "States it.", "weight": True}],
            },
            "weight must be a number",
        ),
        (
            {
                "rubric_id": "refund-reply",
                "criteria": [
                    {"id": "states_window", "text": "States it.", "weight": 0},
                    {"id": "cites_policy", "text": "Cites it.", "weight": 0},
                ],
            },
            "weight above zero",
        ),
        (
            {
                "rubric_id": "refund-reply",
                "criteria": [
                    {
                        "id": "invents_terms",
                        "text": "Invents a term.",
                        "weight": 2,
                        "polarity": "negative",
                    }
                ],
            },
            "weight above zero",
        ),
        ({"rubric_id": "refund-reply", "criteria": []}, "at least 1 item"),
        (
            {
                "rubric_id": "Refund Reply",
                "criteria": [{"id": "states_window", "text": "States it.", "weight": 1}],
            },
            "string_pattern_mismatch",
        ),
    ],
)
def test_rubric_spec_is_rejected_at_load_time_with_a_clear_error(spec, message):
    with pytest.raises(ValueError, match="Invalid rubric spec"):
        parse_rubric_spec(spec)
    with pytest.raises(ValueError, match=message):
        parse_rubric_spec(spec)


def test_rubric_spec_defaults_polarity_and_reports_the_score_denominator():
    rubric = parse_rubric_spec(RUBRIC)

    assert [criterion.polarity for criterion in rubric.criteria[:3]] == ["positive"] * 3
    assert rubric.positive_weight() == 6


def test_parser_computes_the_hand_computed_weighted_score():
    # Met positive weights 3 + 1 of a possible 3 + 2 + 1, no penalty: 4 / 6.
    result = parse_rubric_response(
        recorded_response(
            states_window=True, cites_policy=False, next_step=True, invents_terms=False
        ),
        RUBRIC,
    )

    assert [item["status"] for item in result["criteria"]] == [
        "met",
        "not_met",
        "met",
        "not_met",
    ]
    assert result["weight_met"] == 4
    assert result["weight_possible"] == 6
    assert result["score"] == pytest.approx(4 / 6)
    assert result["status_counts"] == {"met": 2, "not_met": 2, "unknown": 0}
    assert result["critique"] == "Specific, grounded critique."
    assert result["parse_error"] is None


def test_met_negative_criterion_subtracts_its_weight_and_the_score_clips_at_zero():
    result = parse_rubric_response(
        recorded_response(
            states_window=False, cites_policy=False, next_step=True, invents_terms=True
        ),
        RUBRIC,
    )

    # 1 earned minus the 2-point penalty is negative, and the penalty stays out of the denominator.
    assert result["weight_met"] == -1
    assert result["weight_possible"] == 6
    assert result["score"] == 0.0


def test_missing_and_unparseable_criteria_are_unknown_and_leave_the_denominator():
    result = parse_rubric_response(
        json.dumps(
            {
                "critique": "Only two items were decided.",
                "criteria": [
                    {"id": "states_window", "met": True},
                    {"id": "cites_policy", "met": "probably"},
                    {"id": "next_step", "met": False},
                    {"met": True},
                ],
            }
        ),
        RUBRIC,
    )

    statuses = {item["id"]: item["status"] for item in result["criteria"]}
    assert statuses == {
        "states_window": "met",
        "cites_policy": "unknown",
        "next_step": "not_met",
        "invents_terms": "unknown",
    }
    # Unknown items are excluded rather than counted as met: 3 of the resolved 3 + 1.
    assert result["weight_possible"] == 4
    assert result["score"] == pytest.approx(3 / 4)
    assert result["unparsed_entries"] == 2


def test_repeated_and_unexpected_ids_are_reported_without_being_resolved():
    # The contract is one verdict per id, so a repeat is unresolved even when the
    # repeated verdicts agree: it is reported rather than scored on either reading.
    result = parse_rubric_response(
        json.dumps(
            {
                "critique": "Contradicts itself and names an item twice.",
                "criteria": [
                    {"id": "states_window", "met": True},
                    {"id": "states_window", "met": False},
                    {"id": "cites_policy", "met": "yes"},
                    {"id": "cites_policy", "met": True},
                    {"id": "next_step", "met": True},
                    {"id": "invents_terms", "met": False},
                    {"id": "tone", "met": True},
                ],
            }
        ),
        RUBRIC,
    )

    statuses = {item["id"]: item["status"] for item in result["criteria"]}
    assert statuses["states_window"] == "unknown"
    assert statuses["cites_policy"] == "unknown"
    assert result["repeated_ids"] == ["cites_policy", "states_window"]
    assert result["unexpected_ids"] == ["tone"]
    # Only the next-step item was decided once, so it alone carries the score.
    assert result["weight_possible"] == 1
    assert result["score"] == pytest.approx(1.0)


def test_an_id_repeated_with_the_same_verdict_stays_unknown_and_is_reported():
    result = parse_rubric_response(
        json.dumps(
            {
                "critique": "Decides the window item twice the same way.",
                "criteria": [
                    {"id": "states_window", "met": True},
                    {"id": "states_window", "met": True},
                    {"id": "cites_policy", "met": True},
                    {"id": "next_step", "met": True},
                    {"id": "invents_terms", "met": False},
                ],
            }
        ),
        RUBRIC,
    )

    statuses = {item["id"]: item["status"] for item in result["criteria"]}
    assert statuses["states_window"] == "unknown"
    assert result["repeated_ids"] == ["states_window"]
    # The repeated item leaves the denominator instead of being scored on trust.
    assert result["weight_possible"] == 3
    assert result["score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "response",
    ["not json", '["states_window"]', '{"critique": "No checklist."}'],
)
def test_an_unusable_response_leaves_every_criterion_unknown_and_unscored(response):
    result = parse_rubric_response(response, RUBRIC)

    assert {item["status"] for item in result["criteria"]} == {"unknown"}
    assert result["score"] is None
    assert result["weight_possible"] == 0
    assert result["parse_error"]


@pytest.mark.parametrize("critique", [None, "", "   ", 42])
def test_a_response_without_a_critique_is_unusable_and_unscored(critique):
    # The critique is the diagnostic the checklist exists for, so verdicts without one
    # are not scored and stay out of the judge's reuse cache.
    payload = {
        "criteria": [
            {"id": "states_window", "met": True},
            {"id": "cites_policy", "met": True},
            {"id": "next_step", "met": True},
            {"id": "invents_terms", "met": False},
        ]
    }
    if critique is not None:
        payload["critique"] = critique

    result = parse_rubric_response(json.dumps(payload), RUBRIC)

    assert result["critique"] is None
    assert result["parse_error"] == "Rubric judge response has no critique"
    assert {item["status"] for item in result["criteria"]} == {"unknown"}
    assert result["score"] is None


def test_checklist_prompt_section_lists_every_id_weight_and_penalty():
    checklist = render_rubric_checklist(RUBRIC)

    assert "- states_window (weight 3): States the refund window." in checklist
    assert (
        "- invents_terms (weight 2) [counts against the response when met]: "
        "Invents a refund term." in checklist
    )


def dataset_results():
    """Score four recorded judge responses against the rubric."""
    responses = [
        recorded_response(
            states_window=True, cites_policy=True, next_step=True, invents_terms=False
        ),
        recorded_response(
            states_window=True, cites_policy=False, next_step=True, invents_terms=False
        ),
        recorded_response(
            states_window=False, cites_policy=False, next_step=True, invents_terms=True
        ),
        recorded_response(states_window=True, cites_policy=False, invents_terms=False),
    ]
    return [parse_rubric_response(response, RUBRIC) for response in responses]


def test_aggregate_report_ranks_criteria_by_dataset_failure_rate():
    report = RubricReport.model_validate(aggregate_rubric_results(RUBRIC, dataset_results()))

    assert [criterion.id for criterion in report.criteria] == [
        "cites_policy",
        "states_window",
        "invents_terms",
        "next_step",
    ]
    rates = {criterion.id: criterion.failure_rate for criterion in report.criteria}
    assert rates["cites_policy"] == pytest.approx(0.75)
    # A met negative criterion is the failure it names.
    assert rates["invents_terms"] == pytest.approx(0.25)
    assert rates["next_step"] == 0.0
    undecided = next(item for item in report.criteria if item.id == "next_step")
    assert undecided.unknown == 1
    assert undecided.evaluated == 3
    assert undecided.unknown_rate == pytest.approx(0.25)
    assert report.responses == report.scored_responses == 4
    assert report.mean_score == pytest.approx((1 + 4 / 6 + 0 + 0.6) / 4)


def test_criteria_no_response_resolved_rank_last_without_a_rate():
    results = [parse_rubric_response(recorded_response(states_window=False), RUBRIC)]

    report = RubricReport.model_validate(aggregate_rubric_results(RUBRIC, results))

    assert report.criteria[0].id == "states_window"
    assert [criterion.failure_rate for criterion in report.criteria[1:]] == [None, None, None]
    assert [criterion.id for criterion in report.criteria[1:]] == [
        "cites_policy",
        "invents_terms",
        "next_step",
    ]


def test_aggregate_report_is_byte_identical_across_two_runs():
    first = json.dumps(aggregate_rubric_results(RUBRIC, dataset_results()), indent=2)
    second = json.dumps(aggregate_rubric_results(RUBRIC, dataset_results()), indent=2)

    assert first == second


def test_unscored_responses_are_counted_separately_from_the_mean():
    results = [*dataset_results(), parse_rubric_response("not json", RUBRIC)]

    report = RubricReport.model_validate(aggregate_rubric_results(RUBRIC, results))

    assert report.responses == 5
    assert report.scored_responses == 4
    assert report.unscored_responses == 1
    assert report.unparseable_responses == 1
    assert report.mean_score == pytest.approx((1 + 4 / 6 + 0 + 0.6) / 4)


def test_aggregate_rejects_results_from_another_rubric():
    other = {
        "rubric_id": "other-rubric",
        "criteria": [{"id": "tone", "text": "Stays polite.", "weight": 1}],
    }
    results = [parse_rubric_response(recorded_response(tone=True), other)]

    with pytest.raises(ValueError, match="scored against rubric 'other-rubric'"):
        aggregate_rubric_results(RUBRIC, results)


def test_aggregate_rejects_a_criterion_outside_the_rubric():
    result = dict(dataset_results()[0])
    result["criteria"] = [
        *result["criteria"],
        {"id": "tone", "status": "met", "weight": 1, "polarity": "positive"},
    ]

    with pytest.raises(ValueError, match="not part of rubric refund-reply"):
        aggregate_rubric_results(RUBRIC, [result])


def test_aggregate_rejects_same_shaped_results_from_another_rubric():
    # Same criterion ids, different weights and an inverted penalty: not comparable.
    restated = {
        "rubric_id": "refund-reply-v2",
        "criteria": [
            {"id": "states_window", "text": "States the refund window.", "weight": 1},
            {"id": "cites_policy", "text": "Cites the policy section.", "weight": 1},
            {"id": "next_step", "text": "Gives the next concrete step.", "weight": 1},
            {"id": "invents_terms", "text": "Invents a refund term.", "weight": 5},
        ],
    }
    results = [
        parse_rubric_response(
            recorded_response(
                states_window=True, cites_policy=False, next_step=True, invents_terms=True
            ),
            restated,
        )
    ]

    with pytest.raises(ValueError, match="scored against rubric 'refund-reply-v2'"):
        aggregate_rubric_results(RUBRIC, results)


def test_aggregate_rejects_a_result_that_omits_a_rubric_criterion():
    # An omitted criterion would still raise `responses`, understating unknown coverage.
    result = dict(dataset_results()[0])
    result["criteria"] = [item for item in result["criteria"] if item["id"] != "next_step"]

    with pytest.raises(ValueError, match="omits rubric criteria: next_step"):
        aggregate_rubric_results(RUBRIC, [result])


def test_aggregate_rejects_a_result_that_repeats_a_criterion():
    result = dict(dataset_results()[0])
    states_window = next(item for item in result["criteria"] if item["id"] == "states_window")
    result["criteria"] = [*result["criteria"], states_window]

    with pytest.raises(ValueError, match="reports criterion 'states_window' more than once"):
        aggregate_rubric_results(RUBRIC, [result])


def test_aggregate_names_the_result_row_that_is_incomplete():
    results = [*dataset_results()]
    results[2] = dict(results[2])
    results[2]["criteria"] = []

    with pytest.raises(ValueError, match="Result 2 omits rubric criteria"):
        aggregate_rubric_results(RUBRIC, results)


def test_unknown_counts_cover_every_response_in_the_report():
    report = RubricReport.model_validate(aggregate_rubric_results(RUBRIC, dataset_results()))

    for criterion in report.criteria:
        assert criterion.evaluated + criterion.unknown == report.responses


@pytest.mark.parametrize("score", [1.0, 0.0, None])
def test_aggregate_rejects_a_score_that_contradicts_the_criterion_verdicts(score):
    # 4 of a possible 6 was actually met; the mean must not restate anything else.
    result = dict(dataset_results()[1])
    result["score"] = score

    with pytest.raises(ValueError, match="verdicts weigh out to"):
        aggregate_rubric_results(RUBRIC, [result])


def test_aggregate_rejects_a_non_numeric_score():
    result = dict(dataset_results()[0])
    result["score"] = "1.0"

    with pytest.raises(ValueError, match="records a non-numeric score"):
        aggregate_rubric_results(RUBRIC, [result])


def test_aggregate_mean_is_recomputed_from_the_reported_verdicts():
    results = dataset_results()
    # A rounded score stays within tolerance, and the mean comes from the verdicts.
    results[1] = {**results[1], "score": round(4 / 6, 12)}

    report = RubricReport.model_validate(aggregate_rubric_results(RUBRIC, results))

    assert report.mean_score == pytest.approx((1 + 4 / 6 + 0 + 0.6) / 4)


def test_aggregate_rejects_a_result_that_records_no_rubric():
    result = dict(dataset_results()[0])
    del result["rubric_id"]

    with pytest.raises(ValueError, match="scored against rubric None"):
        aggregate_rubric_results(RUBRIC, [result])


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
