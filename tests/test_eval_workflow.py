import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from components.eval_workflow import (
    EvaluatorDefinition,
    bootstrap_metric_intervals,
    build_evaluator_definition,
    build_review_bundle,
    corrected_success_rate,
    dataframe_to_records,
    describe_leakage,
    detect_evaluator_leakage,
    estimate_success_rate,
    find_related_records,
    initialize_review_workspace,
    load_review_workspace,
    normalize_trace_segments,
    parse_evaluator_definition,
    save_evaluator_definition,
    select_diverse_samples,
    split_labeled_records,
    summarize_coverage,
    train_split_expert_examples,
    validate_evaluator_definition,
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


def labeled_split_records(count=12):
    return [
        {
            "id": index,
            "input": f"Customer asks about the refund policy for order {index:03d}",
            "output": f"Agent answer {index:03d} about the refund window",
            "human_label": "Fail" if index % 2 else "Pass",
            "critique": f"Critique {index:03d}: the agent invents a policy detail.",
        }
        for index in range(count)
    ]


def train_seeded_evaluator(records):
    return build_evaluator_definition(
        "Unsupported claim",
        {"description": "The answer asserts facts absent from the supplied evidence."},
        input_fields=["input", "output"],
        source_annotation_ids=["train:1", "train:0"],
        expert_examples=train_split_expert_examples(
            records,
            input_fields=["input", "output"],
            label_field="human_label",
            critique_field="critique",
        ),
        errors_reviewed=True,
    )


def test_train_split_examples_are_balanced_and_attributed():
    examples = train_split_expert_examples(
        labeled_split_records(),
        input_fields=["input", "output"],
        label_field="human_label",
        critique_field="critique",
    )

    assert [example["result"] for example in examples] == ["Fail"] * 4 + ["Pass"] * 4
    assert all(set(example["input"]) == {"input", "output"} for example in examples)
    assert [example["source_id"] for example in examples[:4]] == [
        "train:1",
        "train:3",
        "train:5",
        "train:7",
    ]


def test_train_split_examples_skip_records_without_a_critique():
    records = labeled_split_records()
    records[3]["critique"] = "   "

    examples = train_split_expert_examples(
        records,
        input_fields=["input"],
        label_field="human_label",
        critique_field="critique",
    )

    assert [example["source_id"] for example in examples[:4]] == [
        "train:1",
        "train:5",
        "train:7",
        "train:9",
    ]
    with pytest.raises(ValueError, match="critique"):
        train_split_expert_examples(
            records,
            input_fields=["input"],
            label_field="human_label",
            critique_field="expert_note",
        )
    with pytest.raises(ValueError, match="label field"):
        train_split_expert_examples(
            records,
            input_fields=["input"],
            label_field="verdict",
            critique_field="critique",
        )


def test_train_split_examples_reject_records_without_input_content():
    records = labeled_split_records()
    records[1]["input"] = "   "
    records[1]["output"] = None

    with pytest.raises(ValueError, match="record 1 carries no content"):
        train_split_expert_examples(
            records,
            input_fields=["input", "output"],
            label_field="human_label",
            critique_field="critique",
        )
    with pytest.raises(ValueError, match="no content"):
        train_split_expert_examples(
            labeled_split_records(),
            input_fields=["question"],
            label_field="human_label",
            critique_field="critique",
        )
    with pytest.raises(ValueError, match="one input field is required"):
        train_split_expert_examples(
            labeled_split_records(),
            input_fields=[],
            label_field="human_label",
            critique_field="critique",
        )


def test_train_split_examples_keep_records_missing_an_optional_field():
    records = labeled_split_records()
    for record in records:
        record.pop("output")

    examples = train_split_expert_examples(
        records,
        input_fields=["input", "output"],
        label_field="human_label",
        critique_field="critique",
    )

    assert all(set(example["input"]) == {"input"} for example in examples)


def test_train_split_examples_require_both_verdicts():
    records = labeled_split_records()
    for record in records:
        if record["human_label"] == "Pass":
            record["critique"] = ""

    with pytest.raises(ValueError, match="one critiqued Pass record"):
        train_split_expert_examples(
            records,
            input_fields=["input"],
            label_field="human_label",
            critique_field="critique",
        )


def test_leakage_check_reports_every_held_out_record_in_the_definition():
    records = labeled_split_records()
    evaluator = train_seeded_evaluator(records)
    leaked = dict(records[1])
    leaked["input"] = f"   {leaked['input'].upper()}\n "

    report = detect_evaluator_leakage(evaluator, {"dev": [records[8]], "test": [leaked]})

    assert report["clean"] is False
    assert report["leaked_records"] == 1
    assert {match["split"] for match in report["matches"]} == {"test"}
    assert {match["field"] for match in report["matches"]} == {"input", "output"}
    assert report["matches"][0]["location"] == "expert_examples[1].input"
    assert report["matches"][0]["record_id"] == "1"
    assert "test[0].input in expert_examples[1].input" in describe_leakage(report)


def test_leakage_check_passes_when_examples_come_from_train_only():
    records = labeled_split_records()
    report = detect_evaluator_leakage(
        train_seeded_evaluator(records), {"dev": records[8:10], "test": records[10:]}
    )

    assert report["clean"] is True
    assert report["matches"] == []
    assert report["checked_splits"] == {"dev": 2, "test": 2}
    assert report["records_with_field"] == {"input": 4, "output": 4}
    assert report["compared_texts"] == {"input": 4, "output": 4}


def test_leakage_check_matches_multiline_record_text():
    records = labeled_split_records()
    records[1]["input"] = "Customer asks about the refund policy\nfor order 001"
    evaluator = train_seeded_evaluator(records)

    report = detect_evaluator_leakage(evaluator, {"test": [records[1]]})

    assert report["clean"] is False
    assert report["matches"][0]["field"] == "input"
    assert report["matches"][0]["location"] == "expert_examples[1].input"


def test_leakage_check_matches_quoted_and_backslashed_record_text():
    records = labeled_split_records()
    records[1]["input"] = 'Customer asks "where is my refund" for path C:\\orders\\001'
    evaluator = train_seeded_evaluator(records)

    report = detect_evaluator_leakage(evaluator, {"test": [records[1]]})

    assert report["clean"] is False
    assert report["matches"][0]["field"] == "input"
    assert report["matches"][0]["location"] == "expert_examples[1].input"


@pytest.mark.parametrize("separator", [" ", "\n", "\t", "\r", "\f", "\v"])
def test_leakage_check_matches_quoted_record_text_inside_the_prompt(separator):
    records = labeled_split_records()
    evaluator = train_seeded_evaluator(records)
    quoted = 'The agent replied "we refund any order"' + separator + 'without evidence'
    evaluator["prompt"] = evaluator["prompt"].replace(
        "## Trace or output to evaluate",
        f"## Pasted trace\n{json.dumps({'output': quoted}, ensure_ascii=False)}\n\n"
        "## Trace or output to evaluate",
    )

    report = detect_evaluator_leakage(evaluator, {"test": [{"id": 99, "output": quoted}]})

    assert report["clean"] is False
    assert report["matches"][0]["location"] == "prompt"


def test_leakage_check_is_not_clean_when_no_compared_field_exists():
    records = labeled_split_records()

    report = detect_evaluator_leakage(
        train_seeded_evaluator(records),
        {"test": [{"prompt": "Customer asks about the refund policy for order 011"}]},
    )

    assert report["records_with_field"] == {"input": 0, "output": 0}
    assert report["fields_without_records"] == ["input", "output"]
    assert report["matches"] == []
    assert report["clean"] is False
    assert "compared no held-out records" in describe_leakage(report)


def test_leakage_check_skips_incidental_short_texts():
    records = labeled_split_records()

    report = detect_evaluator_leakage(
        train_seeded_evaluator(records),
        {"test": [{"input": records[10]["input"], "output": "Fail"}]},
        minimum_characters=8,
    )

    assert report["clean"] is True
    assert report["skipped_short_texts"] == 1
    assert report["compared_texts"] == {"input": 1, "output": 0}


def test_leakage_check_is_not_clean_when_every_text_is_too_short():
    records = labeled_split_records()

    report = detect_evaluator_leakage(
        train_seeded_evaluator(records),
        {"test": [{"input": "Pass", "output": "Fail"}, {"input": None, "output": ""}]},
        minimum_characters=8,
    )

    assert report["records_with_field"] == {"input": 2, "output": 2}
    assert report["compared_texts"] == {"input": 0, "output": 0}
    assert report["fields_without_records"] == []
    assert report["matches"] == []
    assert report["clean"] is False
    assert "compared no held-out text for input, output" in describe_leakage(report)
    assert "shorter than 8 characters" in describe_leakage(report)


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


def test_corrected_success_rate_matches_rogan_gladen():
    assert corrected_success_rate(0.88, 0.9, 0.85) == pytest.approx(0.73 / 0.75)
    assert corrected_success_rate(0.99, 0.9, 0.85) == 1.0
    assert corrected_success_rate(0.05, 0.9, 0.85) == 0.0
    with pytest.raises(ValueError, match="no better than chance"):
        corrected_success_rate(0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="TPR must be between 0 and 1"):
        corrected_success_rate(0.5, 1.2, 0.9)


def test_estimate_success_rate_reports_raw_corrected_and_interval():
    human = ["Pass"] * 40 + ["Fail"] * 40
    judge = ["Pass"] * 36 + ["Fail"] * 4 + ["Fail"] * 34 + ["Pass"] * 6
    population = ["Pass"] * 88 + ["Fail"] * 12

    estimate = estimate_success_rate(population, human, judge, iterations=2000, seed=7)

    assert estimate["observed_pass_rate"] == 0.88
    assert estimate["judge_tpr"] == 0.9
    assert estimate["judge_tnr"] == 0.85
    assert estimate["corrected_pass_rate"] == pytest.approx(0.73 / 0.75)
    low, high = estimate["confidence_interval_95"]
    assert 0 <= low < estimate["corrected_pass_rate"] <= high <= 1
    assert estimate["bootstrap_samples_uncorrectable"] == 0


def test_uncorrectable_resamples_widen_the_interval():
    # A judge barely above chance: many resamples leave TPR + TNR <= 1.
    human = ["Pass"] * 10 + ["Fail"] * 10
    judge = ["Pass"] * 6 + ["Fail"] * 4 + ["Fail"] * 5 + ["Pass"] * 5
    population = ["Pass"] * 70 + ["Fail"] * 30

    estimate = estimate_success_rate(population, human, judge, iterations=2000, seed=3)

    assert estimate["bootstrap_samples_uncorrectable"] > 50
    assert estimate["confidence_interval_95"] == [0.0, 1.0]


def test_estimate_bootstrap_is_chunked_and_deterministic(monkeypatch):
    from components.eval_workflow import validation

    human = ["Pass"] * 40 + ["Fail"] * 40
    judge = ["Pass"] * 36 + ["Fail"] * 4 + ["Fail"] * 34 + ["Pass"] * 6
    population = ["Pass"] * 88 + ["Fail"] * 12
    whole = estimate_success_rate(population, human, judge, iterations=500, seed=11)
    monkeypatch.setattr(validation, "_BOOTSTRAP_CELLS_PER_CHUNK", 80 * 7)
    chunked = estimate_success_rate(population, human, judge, iterations=500, seed=11)

    assert chunked["bootstrap_iterations"] == 500
    assert 0 <= chunked["confidence_interval_95"][0] <= chunked["confidence_interval_95"][1] <= 1
    assert whole["corrected_pass_rate"] == chunked["corrected_pass_rate"]


def test_estimate_requires_positive_bootstrap_iterations():
    with pytest.raises(ValueError, match="positive"):
        estimate_success_rate(["Pass"], ["Pass", "Fail"], ["Pass", "Fail"], iterations=0)


def test_estimate_success_rate_requires_both_test_classes():
    with pytest.raises(ValueError, match="both Pass and Fail"):
        estimate_success_rate(["Pass"], ["Pass", "Pass"], ["Pass", "Fail"])


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


def test_labeled_splits_keep_every_partition_nonempty_at_extreme_valid_fractions():
    records = [
        {"id": index, "human_label": label}
        for label in ("Pass", "Fail")
        for index in range(3)
    ]

    splits = split_labeled_records(
        records,
        "human_label",
        train_fraction=0.49,
        dev_fraction=0.49,
    )

    assert {name: len(items) for name, items in splits.items()} == {
        "train": 2,
        "dev": 2,
        "test": 2,
    }
    assert all({item["human_label"] for item in items} == {"Pass", "Fail"} for items in splits.values())


def test_evaluator_definition_validation_rejects_shallow_imports():
    with pytest.raises(ValueError, match="schema_version"):
        validate_evaluator_definition(
            {"kind": "llm_judge", "prompt": "{{evaluation_input}}"},
            required_kind="llm_judge",
        )


def test_evaluator_definition_uses_a_strict_typed_contract():
    evaluator = build_evaluator_definition(
        "Unsupported claim",
        {"description": "The output contains an unsupported claim."},
    )

    parsed = parse_evaluator_definition(evaluator)

    assert isinstance(parsed, EvaluatorDefinition)
    evaluator["validation"]["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        parse_evaluator_definition(evaluator)


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


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
