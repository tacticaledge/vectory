import json

import pandas as pd
import pytest

from components.eval_workflow import (
    bootstrap_metric_intervals,
    build_evaluator_definition,
    build_review_bundle,
    dataframe_to_records,
    find_related_records,
    initialize_review_workspace,
    load_review_workspace,
    normalize_trace_segments,
    save_evaluator_definition,
    select_diverse_samples,
    split_labeled_records,
    summarize_coverage,
    validate_evaluator_labels,
)


def sample_records(count=24):
    topics = ("billing refund", "account login", "shipping delay", "tool timeout")
    return [
        {
            "id": index,
            "input": f"Question about {topics[index % len(topics)]} case {index}",
            "output": f"Answer {index % 7}",
            "trace": [{"role": "assistant", "content": f"Step {index}"}],
        }
        for index in range(count)
    ]


def test_dataframe_records_are_json_safe():
    records = dataframe_to_records(
        pd.DataFrame([{"value": float("nan"), "missing": pd.NA, "timestamp": pd.Timestamp("2026-01-02")}])
    )

    assert records == [{"value": None, "missing": None, "timestamp": "2026-01-02T00:00:00"}]
    json.dumps(records)


def test_diverse_sampling_is_deterministic_and_mixed():
    records = sample_records()

    first = select_diverse_samples(records, sample_size=12, seed=7)
    second = select_diverse_samples(records, sample_size=12, seed=7)

    assert first == second
    assert len(first) == 12
    assert len({item["index"] for item in first}) == 12
    assert {item["strategy"] for item in first} == {
        "cluster_representative",
        "random_exploration",
    }


def test_dimension_coverage_prioritizes_rare_product_slices():
    records = [
        {"input": f"Common {index}", "output": "ok", "scenario": "common"}
        for index in range(12)
    ] + [
        {"input": "Rare path", "output": "failed", "scenario": "rare"},
    ]

    selected = select_diverse_samples(
        records,
        sample_size=6,
        dimension_fields=["scenario"],
    )

    assert 12 in {item["index"] for item in selected}
    assert any(item["strategy"] == "dimension_coverage" for item in selected)


def test_related_examples_exclude_already_reviewed_records():
    records = sample_records()
    related = find_related_records(
        records,
        ["shipping delay and late package"],
        exclude_indices=[2, 6],
        limit=5,
    )

    assert len(related) == 5
    assert {item["index"] for item in related}.isdisjoint({2, 6})
    assert related == sorted(related, key=lambda item: (-item["score"], item["index"]))


def test_coverage_never_claims_prevalence():
    samples = select_diverse_samples(sample_records(), sample_size=8)
    report = summarize_coverage(samples, [item["index"] for item in samples[:3]])

    assert report["reviewed_selected_count"] == 3
    assert "must not be used" in report["prevalence_warning"]


def test_review_workspace_is_portable_and_versioned(tmp_path):
    records = sample_records(8)
    samples = select_diverse_samples(records, sample_size=4)
    workspace = tmp_path / "review"

    manifest = initialize_review_workspace(
        workspace,
        records,
        samples,
        source="traces.jsonl",
        reviewer={"name": "domain-expert", "role": "Support Director"},
        dimension_fields=["feature", "scenario"],
    )
    loaded = load_review_workspace(workspace)

    assert manifest["schema_version"] == "1.0"
    assert manifest["sampling"]["unbiased_prevalence_estimate"] is False
    assert manifest["review_protocol"]["critique_required"] is True
    assert manifest["review_protocol"]["principal_domain_expert"]["role"] == "Support Director"
    assert manifest["review_protocol"]["dimension_fields"] == ["feature", "scenario"]
    assert loaded["records"] == records
    assert loaded["samples"] == samples
    assert (workspace / "annotations.json").is_file()
    assert (workspace / "taxonomy.json").is_file()


def test_review_bundle_preserves_human_artifacts():
    bundle = build_review_bundle(
        sample_records(2),
        [{"index": 0, "cluster": 0}],
        annotations={0: {"open_code": "Invented a policy", "pass_fail": False}},
        taxonomy={"Fabrication": {"description": "Claims unsupported policy facts"}},
        evaluators={"failure-fabrication": {"status": "draft_unvalidated"}},
    )

    assert bundle["annotations"]["0"]["open_code"] == "Invented a policy"
    assert bundle["taxonomy"]["Fabrication"]["description"]
    assert bundle["evaluators"]["failure-fabrication"]["status"] == "draft_unvalidated"
    assert bundle["sampling"]["unbiased_prevalence_estimate"] is False


def test_promoted_evaluator_is_binary_and_unvalidated(tmp_path):
    workspace = tmp_path / "review"
    initialize_review_workspace(workspace, sample_records(2), [{"index": 0, "cluster": 0}])
    evaluator = build_evaluator_definition(
        "Unsupported claim",
        {
            "description": "The answer asserts facts absent from the supplied evidence.",
            "examples": ["The refund policy says 90 days, but no policy was supplied."],
        },
        source_annotation_ids=[1, 2],
        expert_examples=[
            {
                "input": {"input": "What is the policy?", "output": "It is 90 days."},
                "critique": "The output invents the 90-day period.",
                "result": "Fail",
            },
            {
                "input": {"input": "What is the policy?", "output": "No policy was supplied."},
                "critique": "The response correctly avoids an unsupported claim.",
                "result": "Pass",
            },
        ],
        errors_reviewed=True,
    )
    path = save_evaluator_definition(workspace, evaluator)

    assert evaluator["decision"]["type"] == "binary"
    assert evaluator["status"] == "draft_unvalidated"
    assert evaluator["validation"]["required"] is True
    assert evaluator["output_schema"]["properties"]["result"]["enum"] == ["Pass", "Fail"]
    assert "critique" in evaluator["output_schema"]["required"]
    assert evaluator["lifecycle_checkpoints"]["obvious_errors_reviewed"] is True
    assert len(evaluator["expert_examples"]) == 2
    assert "The output invents the 90-day period" in evaluator["prompt"]
    assert path.is_file()


def test_rule_promotion_requires_a_pattern():
    with pytest.raises(ValueError, match="require a rule pattern"):
        build_evaluator_definition("Leak", {"description": "Contains a secret"}, kind="regex")


def test_validation_reports_both_sides_of_binary_classifier():
    report = validate_evaluator_labels(
        ["Pass", "Pass", "Fail", "Fail"],
        ["Pass", "Fail", "Fail", "Pass"],
    )

    assert report["tpr"] == 0.5
    assert report["tnr"] == 0.5
    assert report["balanced_accuracy"] == 0.5
    assert report["per_class"]["Pass"]["precision"] == 0.5
    assert report["per_class"]["Fail"]["recall"] == 0.5
    assert report["confusion_matrix"] == {
        "true_pass": 1,
        "false_fail": 1,
        "true_fail": 1,
        "false_pass": 1,
    }


def test_bootstrap_intervals_are_bounded():
    labels = ["Pass", "Pass", "Pass", "Fail", "Fail", "Fail"]
    intervals = bootstrap_metric_intervals(labels, labels, iterations=100, seed=3)

    assert intervals == {"tpr": [1.0, 1.0], "tnr": [1.0, 1.0]}


def test_labeled_splits_are_disjoint_and_stratified():
    records = [
        {"id": index, "human_label": "Pass" if index % 2 == 0 else "Fail"}
        for index in range(40)
    ]
    splits = split_labeled_records(records, "human_label", seed=9)

    identifiers = [{item["id"] for item in splits[name]} for name in ("train", "dev", "test")]
    assert len(set.union(*identifiers)) == 40
    assert not identifiers[0] & identifiers[1]
    assert not identifiers[0] & identifiers[2]
    assert not identifiers[1] & identifiers[2]
    for split in splits.values():
        assert {item["human_label"] for item in split} == {"Pass", "Fail"}


def test_labeled_splits_require_enough_examples_for_all_three_sets():
    with pytest.raises(ValueError, match="at least three Pass and three Fail"):
        split_labeled_records(
            [
                {"id": 1, "human_label": "Pass"},
                {"id": 2, "human_label": "Pass"},
                {"id": 3, "human_label": "Fail"},
                {"id": 4, "human_label": "Fail"},
            ],
            "human_label",
        )


def test_trace_normalization_preserves_order_and_details():
    segments = normalize_trace_segments(
        {
            "events": json.dumps(
                [
                    {"role": "user", "content": "Find the invoice"},
                    {"type": "tool_call", "tool_name": "search", "arguments": {"q": "invoice"}},
                    {"type": "tool_result", "output": {"id": 12}},
                ]
            )
        }
    )

    assert [segment["index"] for segment in segments] == [0, 1, 2]
    assert segments[1]["title"] == "search"
    assert segments[1]["details"] == {"arguments": {"q": "invoice"}}
    assert '"id": 12' in segments[2]["content"]
