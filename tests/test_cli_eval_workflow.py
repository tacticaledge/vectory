import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from components import __version__ as component_version
from components.eval_workflow import build_evaluator_definition
from vectory_cli import __version__ as cli_version
from vectory_cli.cli import main


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_cli_version_uses_authoritative_component_version(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    assert error.value.code == 0
    assert cli_version == component_version == "1.2.0"
    assert capsys.readouterr().out.strip() == f"Vectory CLI {component_version}"


def test_discover_and_promote_workflow(tmp_path):
    dataset = tmp_path / "traces.jsonl"
    write_jsonl(
        dataset,
        [
            {"input": f"Question {index}", "output": f"Answer {index % 3}"}
            for index in range(12)
        ],
    )
    workspace = tmp_path / "review"

    assert main(["discover", str(dataset), "--workspace", str(workspace), "--sample-size", "6"]) == 0
    taxonomy = {
        "Unsupported claim": {
            "description": "The output makes a factual claim with no supplied evidence.",
            "examples": ["Claims a 90-day policy without a source."],
        }
    }
    (workspace / "taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")
    annotations = {
        "0": {
            "pass_fail": False,
            "failure_modes": ["Unsupported claim"],
            "critique": "The answer invents a policy.",
        },
        "1": {
            "pass_fail": True,
            "failure_modes": [],
            "critique": "The answer is grounded in the supplied information.",
        },
    }
    (workspace / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")

    assert main(["promote", str(workspace), "Unsupported claim", "--errors-reviewed"]) == 0
    evaluator = json.loads(
        (workspace / "evaluators" / "failure-unsupported-claim.json").read_text(encoding="utf-8")
    )
    assert evaluator["status"] == "draft_unvalidated"
    assert evaluator["decision"]["type"] == "binary"
    assert evaluator["lifecycle_checkpoints"]["obvious_errors_reviewed"] is True
    assert {example["result"] for example in evaluator["expert_examples"]} == {"Pass", "Fail"}


def test_force_discover_archives_all_human_review_artifacts(tmp_path):
    dataset = tmp_path / "traces.jsonl"
    write_jsonl(dataset, [{"input": f"Q{index}", "output": f"A{index}"} for index in range(8)])
    workspace = tmp_path / "review"
    assert main(["discover", str(dataset), "--workspace", str(workspace)]) == 0
    (workspace / "annotations.json").write_text('{"0":{"critique":"Keep me"}}', encoding="utf-8")
    (workspace / "evaluators" / "stale.json").write_text('{"status":"validated"}', encoding="utf-8")

    assert main(["discover", str(dataset), "--workspace", str(workspace), "--force"]) == 0

    archives = list(tmp_path.glob("review.archive-*"))
    assert len(archives) == 1
    assert json.loads((archives[0] / "annotations.json").read_text(encoding="utf-8"))["0"]["critique"] == "Keep me"
    assert (archives[0] / "evaluators" / "stale.json").is_file()
    assert not (workspace / "evaluators" / "stale.json").exists()
    assert json.loads((workspace / "annotations.json").read_text(encoding="utf-8")) == {}


def test_split_and_validate_judge_gate(tmp_path):
    labels = tmp_path / "labels.jsonl"
    records = [
        {
            "id": index,
            "human_label": "Pass" if index % 2 == 0 else "Fail",
            "judge_label": "preliminary",
            "scenario": "common" if index % 4 < 2 else "edge",
        }
        for index in range(40)
    ]
    write_jsonl(labels, records)
    split_dir = tmp_path / "splits"

    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human_label",
            "--judge-column",
            "judge_label",
            "--out",
            str(split_dir),
        ]
    ) == 0
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert sum(manifest["counts"].values()) == 40
    assert all((split_dir / f"{name}.jsonl").is_file() for name in ("train", "dev", "test"))
    test_records = [
        json.loads(line) for line in (split_dir / "test.jsonl").read_text().splitlines()
    ]
    for record in test_records:
        record["judge_label"] = record["human_label"]
    write_jsonl(split_dir / "test.jsonl", test_records)

    report = tmp_path / "validation.json"
    assert main(
        [
            "validate-judge",
            str(split_dir / "test.jsonl"),
            "--human-column",
            "human_label",
            "--judge-column",
            "judge_label",
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--bootstrap-iterations",
            "100",
            "--out",
            str(report),
            "--group-by",
            "scenario",
        ]
    ) == 0
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert report_payload["gate_passed"] is True
    assert len(report_payload["groups"]) == 2
    assert report_payload["dataset_role"] == "held_out_test"
    assert report_payload["provenance"]["verified"] is True
    assert report_payload["provenance"]["excluded_prediction_field"] == "judge_label"
    assert "judge_label" not in report_payload["provenance"]["verified_fields"]


def _text_split_fixture(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {
                "id": index,
                "input": f"Customer asks about the refund policy for order {index:03d}",
                "output": f"Agent answer {index:03d} about the refund window",
                "human_label": "Pass" if index % 2 == 0 else "Fail",
                "judge_label": "",
                "critique": f"Critique {index:03d}: the agent invents a policy detail.",
            }
            for index in range(40)
        ],
    )
    split_dir = tmp_path / "splits"
    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human_label",
            "--judge-column",
            "judge_label",
            "--out",
            str(split_dir),
        ]
    ) == 0
    test_path = split_dir / "test.jsonl"
    test_records = [json.loads(line) for line in test_path.read_text().splitlines()]
    for record in test_records:
        record["judge_label"] = record["human_label"]
    write_jsonl(test_path, test_records)
    workspace = tmp_path / "review"
    assert main(["discover", str(labels), "--workspace", str(workspace), "--sample-size", "6"]) == 0
    (workspace / "taxonomy.json").write_text(
        json.dumps(
            {
                "Unsupported claim": {
                    "description": "The answer asserts facts absent from the supplied evidence."
                }
            }
        ),
        encoding="utf-8",
    )
    return split_dir, test_records, workspace


def test_promote_draws_examples_from_the_train_split(tmp_path):
    split_dir, _, workspace = _text_split_fixture(tmp_path)
    train_records = [
        json.loads(line) for line in (split_dir / "train.jsonl").read_text().splitlines()
    ]

    assert main(
        [
            "promote",
            str(workspace),
            "Unsupported claim",
            "--errors-reviewed",
            "--train-split",
            str(split_dir / "train.jsonl"),
            "--input-fields",
            "input",
            "output",
        ]
    ) == 0

    evaluator = json.loads(
        (workspace / "evaluators" / "failure-unsupported-claim.json").read_text(encoding="utf-8")
    )
    train_inputs = {record["input"] for record in train_records}
    assert {example["input"]["input"] for example in evaluator["expert_examples"]} <= train_inputs
    assert {example["result"] for example in evaluator["expert_examples"]} == {"Pass", "Fail"}
    assert all(
        source.startswith("train:")
        for source in evaluator["failure_mode"]["source_annotation_ids"]
    )
    assert main(
        [
            "check-leakage",
            str(workspace / "evaluators" / "failure-unsupported-claim.json"),
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
        ]
    ) == 0


def test_promote_rejects_a_single_verdict_train_split(tmp_path):
    split_dir, _, workspace = _text_split_fixture(tmp_path)
    train_path = split_dir / "train.jsonl"
    train_records = [json.loads(line) for line in train_path.read_text().splitlines()]
    for record in train_records:
        if record["human_label"] == "Pass":
            record["critique"] = ""
    write_jsonl(train_path, train_records)

    with pytest.raises(SystemExit):
        main(
            [
                "promote",
                str(workspace),
                "Unsupported claim",
                "--errors-reviewed",
                "--train-split",
                str(train_path),
                "--input-fields",
                "input",
                "output",
            ]
        )

    assert not (workspace / "evaluators" / "failure-unsupported-claim.json").exists()


def test_promote_rejects_a_train_split_without_the_requested_input_fields(tmp_path):
    split_dir, _, workspace = _text_split_fixture(tmp_path)

    with pytest.raises(SystemExit):
        main(
            [
                "promote",
                str(workspace),
                "Unsupported claim",
                "--errors-reviewed",
                "--train-split",
                str(split_dir / "train.jsonl"),
                "--input-fields",
                "question",
                "context",
            ]
        )

    assert not (workspace / "evaluators" / "failure-unsupported-claim.json").exists()


def test_check_leakage_rejects_a_held_out_split_that_lost_records(tmp_path):
    split_dir, test_records, workspace = _text_split_fixture(tmp_path)
    evaluator_path = workspace / "evaluators" / "leaky.json"
    evaluator_path.write_text(
        json.dumps(
            build_evaluator_definition(
                "Unsupported claim",
                {"description": "The answer asserts facts absent from the supplied evidence."},
                input_fields=["input", "output"],
                source_annotation_ids=["0", "1"],
                expert_examples=[
                    {
                        "input": {"input": test_records[0]["input"]},
                        "critique": "The answer invents a refund window.",
                        "result": "Fail",
                    },
                    {
                        "input": {"input": "Where can I see my past orders?"},
                        "critique": "The answer stays inside the supplied evidence.",
                        "result": "Pass",
                    },
                ],
                errors_reviewed=True,
            )
        ),
        encoding="utf-8",
    )
    # Dropping the leaked record would otherwise hide the leak behind a clean report.
    write_jsonl(split_dir / "test.jsonl", test_records[1:])

    with pytest.raises(SystemExit):
        main(
            [
                "check-leakage",
                str(evaluator_path),
                "--split-manifest",
                str(split_dir / "split_manifest.json"),
                "--splits",
                "test",
            ]
        )


def test_check_leakage_rejects_an_edited_held_out_record(tmp_path):
    split_dir, _, workspace = _text_split_fixture(tmp_path)
    assert main(
        [
            "promote",
            str(workspace),
            "Unsupported claim",
            "--errors-reviewed",
            "--train-split",
            str(split_dir / "train.jsonl"),
            "--input-fields",
            "input",
            "output",
        ]
    ) == 0
    evaluator_path = workspace / "evaluators" / "failure-unsupported-claim.json"
    dev_path = split_dir / "dev.jsonl"
    dev_records = [json.loads(line) for line in dev_path.read_text().splitlines()]
    # The judge prediction column stays mutable, so filling it keeps the split checkable.
    for record in dev_records:
        record["judge_label"] = record["human_label"]
    write_jsonl(dev_path, dev_records)
    assert main(
        [
            "check-leakage",
            str(evaluator_path),
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--splits",
            "dev",
        ]
    ) == 0

    dev_records[0]["input"] = "Replaced held-out question about the refund policy"
    write_jsonl(dev_path, dev_records)

    with pytest.raises(SystemExit):
        main(
            [
                "check-leakage",
                str(evaluator_path),
                "--split-manifest",
                str(split_dir / "split_manifest.json"),
                "--splits",
                "dev",
            ]
        )


def test_check_leakage_reports_held_out_records_in_an_evaluator(tmp_path, capsys):
    split_dir, test_records, workspace = _text_split_fixture(tmp_path)
    evaluator_path = workspace / "evaluators" / "leaky.json"
    evaluator_path.write_text(
        json.dumps(
            build_evaluator_definition(
                "Unsupported claim",
                {"description": "The answer asserts facts absent from the supplied evidence."},
                input_fields=["input", "output"],
                source_annotation_ids=["0", "1"],
                expert_examples=[
                    {
                        "input": {"input": f"   {test_records[0]['input'].upper()}  "},
                        "critique": "The answer invents a refund window.",
                        "result": "Fail",
                    },
                    {
                        "input": {"input": "Where can I see my past orders?"},
                        "critique": "The answer stays inside the supplied evidence.",
                        "result": "Pass",
                    },
                ],
                errors_reviewed=True,
            )
        ),
        encoding="utf-8",
    )
    report = tmp_path / "leakage.json"

    assert main(
        [
            "check-leakage",
            str(evaluator_path),
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--splits",
            "test",
            "--out",
            str(report),
        ]
    ) == 1

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["clean"] is False
    assert payload["leaked_records"] == 1
    assert payload["matches"][0]["split"] == "test"
    assert payload["matches"][0]["record_index"] == 0
    assert payload["matches"][0]["location"] == "expert_examples[1].input"
    assert "Leak: test[0] field input" in capsys.readouterr().out

    for split_name in ("holdout", "train"):
        with pytest.raises(SystemExit):
            main(
                [
                    "check-leakage",
                    str(evaluator_path),
                    "--split-manifest",
                    str(split_dir / "split_manifest.json"),
                    "--splits",
                    split_name,
                ]
            )


def test_check_leakage_fails_when_every_compared_text_is_too_short(tmp_path, capsys):
    split_dir, _, workspace = _text_split_fixture(tmp_path)
    assert main(
        [
            "promote",
            str(workspace),
            "Unsupported claim",
            "--errors-reviewed",
            "--train-split",
            str(split_dir / "train.jsonl"),
            "--input-fields",
            "input",
            "output",
        ]
    ) == 0
    evaluator_path = workspace / "evaluators" / "failure-unsupported-claim.json"
    report = tmp_path / "leakage.json"

    assert main(
        [
            "check-leakage",
            str(evaluator_path),
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--text-fields",
            "human_label",
            "--out",
            str(report),
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "Fields absent from every record" not in output
    assert "Leakage: UNCHECKED" in output
    assert "compared no held-out text for human_label" in output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["clean"] is False
    assert payload["matches"] == []
    assert payload["fields_without_records"] == []
    assert payload["compared_texts"] == {"human_label": 0}
    assert payload["skipped_short_texts"] == payload["records_with_field"]["human_label"]


def test_check_leakage_fails_when_no_compared_field_exists(tmp_path, capsys):
    split_dir, _, workspace = _text_split_fixture(tmp_path)
    assert main(
        [
            "promote",
            str(workspace),
            "Unsupported claim",
            "--errors-reviewed",
            "--train-split",
            str(split_dir / "train.jsonl"),
            "--input-fields",
            "input",
            "output",
        ]
    ) == 0
    evaluator_path = workspace / "evaluators" / "failure-unsupported-claim.json"
    report = tmp_path / "leakage.json"

    assert main(
        [
            "check-leakage",
            str(evaluator_path),
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--text-fields",
            "question",
            "--out",
            str(report),
        ]
    ) == 1

    output = capsys.readouterr().out
    assert "Fields absent from every record: question" in output
    assert "Leakage: UNCHECKED" in output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["clean"] is False
    assert payload["matches"] == []
    assert payload["fields_without_records"] == ["question"]


def test_validate_judge_blocks_a_leakage_check_that_compared_nothing(tmp_path, capsys):
    split_dir, _, workspace = _text_split_fixture(tmp_path)
    assert main(
        [
            "promote",
            str(workspace),
            "Unsupported claim",
            "--errors-reviewed",
            "--train-split",
            str(split_dir / "train.jsonl"),
            "--input-fields",
            "input",
            "output",
        ]
    ) == 0
    report = tmp_path / "validation.json"

    assert main(
        [
            "validate-judge",
            str(split_dir / "test.jsonl"),
            "--human-column",
            "human_label",
            "--judge-column",
            "judge_label",
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--bootstrap-iterations",
            "50",
            "--evaluator",
            str(workspace / "evaluators" / "failure-unsupported-claim.json"),
            "--text-fields",
            "question",
            "--out",
            str(report),
        ]
    ) == 1

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["gate_passed"] is False
    assert payload["leakage"]["clean"] is False
    assert payload["gate_blockers"] == [
        "leakage check compared no held-out records for question: "
        "point --text-fields at fields these records carry"
    ]
    assert "Blocker: leakage check compared no held-out records" in capsys.readouterr().out


def test_validate_judge_refuses_an_evaluator_that_saw_the_test_split(tmp_path, capsys):
    split_dir, test_records, workspace = _text_split_fixture(tmp_path)
    evaluator_path = workspace / "evaluators" / "leaky.json"
    evaluator_path.write_text(
        json.dumps(
            build_evaluator_definition(
                "Unsupported claim",
                {"description": "The answer asserts facts absent from the supplied evidence."},
                input_fields=["input", "output"],
                source_annotation_ids=["0", "1"],
                expert_examples=[
                    {
                        "input": {"input": test_records[1]["input"]},
                        "critique": "The answer invents a refund window.",
                        "result": "Fail",
                    },
                    {
                        "input": {"input": "Where can I see my past orders?"},
                        "critique": "The answer stays inside the supplied evidence.",
                        "result": "Pass",
                    },
                ],
                errors_reviewed=True,
            )
        ),
        encoding="utf-8",
    )
    report = tmp_path / "validation.json"
    arguments = [
        "validate-judge",
        str(split_dir / "test.jsonl"),
        "--human-column",
        "human_label",
        "--judge-column",
        "judge_label",
        "--split-manifest",
        str(split_dir / "split_manifest.json"),
        "--bootstrap-iterations",
        "50",
        "--evaluator",
        str(evaluator_path),
        "--out",
        str(report),
    ]

    assert main(arguments) == 1

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["metrics"]["tpr"] == 1.0
    assert payload["gate_passed"] is False
    assert payload["leakage"]["clean"] is False
    assert payload["leakage"]["leaked_records"] == 1
    assert any("reuses text" in blocker for blocker in payload["gate_blockers"])
    assert "Blocker: evaluator definition reuses text" in capsys.readouterr().out
    assert json.loads(evaluator_path.read_text(encoding="utf-8"))["status"] == "validation_failed"


def test_validate_judge_accepts_an_evaluator_seeded_from_train(tmp_path):
    split_dir, _, workspace = _text_split_fixture(tmp_path)
    assert main(
        [
            "promote",
            str(workspace),
            "Unsupported claim",
            "--errors-reviewed",
            "--train-split",
            str(split_dir / "train.jsonl"),
            "--input-fields",
            "input",
            "output",
        ]
    ) == 0
    evaluator_path = workspace / "evaluators" / "failure-unsupported-claim.json"
    report = tmp_path / "validation.json"

    assert main(
        [
            "validate-judge",
            str(split_dir / "test.jsonl"),
            "--human-column",
            "human_label",
            "--judge-column",
            "judge_label",
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--bootstrap-iterations",
            "50",
            "--evaluator",
            str(evaluator_path),
            "--out",
            str(report),
        ]
    ) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["gate_passed"] is True
    assert payload["gate_blockers"] == []
    assert payload["leakage"]["clean"] is True
    assert payload["leakage"]["records_with_field"]["input"] == len(
        (split_dir / "test.jsonl").read_text().splitlines()
    )


def _validated_judge_fixture(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {
                "id": index,
                "output": f"Agent answer {index:03d} about the refund window",
                "human_label": "Pass" if index % 2 == 0 else "Fail",
                "judge_label": "",
            }
            for index in range(80)
        ],
    )
    split_dir = tmp_path / "splits"
    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human_label",
            "--judge-column",
            "judge_label",
            "--out",
            str(split_dir),
        ]
    ) == 0
    test_path = split_dir / "test.jsonl"
    test_records = [json.loads(line) for line in test_path.read_text().splitlines()]
    for record in test_records:
        record["judge_label"] = record["human_label"]
    write_jsonl(test_path, test_records)
    evaluator_path = tmp_path / "evaluator.json"
    evaluator_path.write_text(
        json.dumps(
            build_evaluator_definition(
                "Unsupported claim",
                {"description": "The output contains an unsupported claim."},
                source_annotation_ids=["pass", "fail"],
                expert_examples=[
                    {"input": {"output": "Grounded"}, "critique": "Grounded answer.", "result": "Pass"},
                    {"input": {"output": "Invented"}, "critique": "Unsupported claim.", "result": "Fail"},
                ],
                errors_reviewed=True,
            )
        ),
        encoding="utf-8",
    )
    production = tmp_path / "production.jsonl"
    write_jsonl(
        production,
        [
            {
                "id": index,
                "verdict": "Pass" if index < 180 else "Fail",
                "evaluator_id": "failure-unsupported-claim",
            }
            for index in range(200)
        ],
    )
    arguments = [
        "estimate",
        str(production),
        "--judge-column",
        "verdict",
        "--test-set",
        str(test_path),
        "--test-judge-column",
        "judge_label",
        "--human-column",
        "human_label",
        "--split-manifest",
        str(split_dir / "split_manifest.json"),
        "--bootstrap-iterations",
        "200",
    ]
    validate_arguments = [
        "validate-judge",
        str(test_path),
        "--human-column",
        "human_label",
        "--judge-column",
        "judge_label",
        "--split-manifest",
        str(split_dir / "split_manifest.json"),
        "--evaluator",
        str(evaluator_path),
        "--bootstrap-iterations",
        "50",
    ]
    return arguments, validate_arguments, evaluator_path, test_path


def test_estimate_gates_on_interval_lower_bound(tmp_path, capsys):
    arguments, validate_arguments, evaluator_path, _ = _validated_judge_fixture(tmp_path)
    assert main(validate_arguments) == 0
    out = tmp_path / "estimate.json"

    assert main(
        [*arguments, "--evaluator", str(evaluator_path), "--min-pass-rate", "0.8", "--out", str(out)]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["estimate"]["observed_pass_rate"] == 0.9
    assert report["estimate"]["corrected_pass_rate"] == pytest.approx(0.9)
    assert report["gate_passed"] is True
    assert report["gate_blockers"] == []
    assert report["provenance"]["verified"] is True
    assert report["evaluator_id"] == "failure-unsupported-claim"
    assert report["dataset_evaluator_column"] == "evaluator_id"
    assert "corrected=0.900" in capsys.readouterr().out

    assert main([*arguments, "--evaluator", str(evaluator_path), "--min-pass-rate", "0.95"]) == 1
    assert main(arguments) == 0


def test_estimate_gate_requires_validated_evaluator(tmp_path):
    arguments, _, evaluator_path, _ = _validated_judge_fixture(tmp_path)
    out = tmp_path / "estimate.json"

    with pytest.raises(SystemExit):
        main([*arguments, "--min-pass-rate", "0.5"])
    assert main(
        [*arguments, "--evaluator", str(evaluator_path), "--min-pass-rate", "0.5", "--out", str(out)]
    ) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate_passed"] is False
    assert report["gate_blockers"] == ["evaluator status is draft_unvalidated, not validated"]


def test_estimate_gate_rejects_labels_that_differ_from_validation(tmp_path):
    arguments, validate_arguments, evaluator_path, test_path = _validated_judge_fixture(tmp_path)
    assert main(validate_arguments) == 0
    test_records = [json.loads(line) for line in test_path.read_text().splitlines()]
    first_fail = next(record for record in test_records if record["human_label"] == "Fail")
    first_fail["judge_label"] = "Pass"
    write_jsonl(test_path, test_records)
    out = tmp_path / "estimate.json"

    assert main(
        [*arguments, "--evaluator", str(evaluator_path), "--min-pass-rate", "0.5", "--out", str(out)]
    ) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate_blockers"] == [
        "test set labels do not match the evaluator's validation report"
    ]


def test_estimate_gate_rejects_reordered_predictions_with_same_totals(tmp_path):
    arguments, validate_arguments, evaluator_path, test_path = _validated_judge_fixture(tmp_path)
    test_records = [json.loads(line) for line in test_path.read_text().splitlines()]
    first, second = [record for record in test_records if record["human_label"] == "Pass"][:2]
    first["judge_label"] = "Fail"
    write_jsonl(test_path, test_records)
    assert main(validate_arguments) == 0
    first["judge_label"] = "Pass"
    second["judge_label"] = "Fail"
    write_jsonl(test_path, test_records)
    out = tmp_path / "estimate.json"

    assert main(
        [*arguments, "--evaluator", str(evaluator_path), "--min-pass-rate", "0.5", "--out", str(out)]
    ) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate_blockers"] == [
        "test set labels do not match the evaluator's validation report"
    ]


def test_estimate_gate_rejects_judged_rows_without_an_evaluator(tmp_path):
    arguments, validate_arguments, evaluator_path, _ = _validated_judge_fixture(tmp_path)
    assert main(validate_arguments) == 0
    production = tmp_path / "production.jsonl"
    records = [json.loads(line) for line in production.read_text().splitlines()]
    records[7].pop("evaluator_id")
    write_jsonl(production, records)
    out = tmp_path / "estimate.json"

    assert main(
        [*arguments, "--evaluator", str(evaluator_path), "--min-pass-rate", "0.5", "--out", str(out)]
    ) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate_passed"] is False
    assert report["gate_blockers"] == [
        "judged dataset does not record 'evaluator_id' on every row, "
        "so its labels cannot be attributed to evaluator failure-unsupported-claim"
    ]


def test_estimate_gate_rejects_labels_from_another_evaluator(tmp_path):
    arguments, validate_arguments, evaluator_path, _ = _validated_judge_fixture(tmp_path)
    assert main(validate_arguments) == 0
    production = tmp_path / "production.jsonl"
    records = [json.loads(line) for line in production.read_text().splitlines()]
    for record in records:
        record["evaluator_id"] = "failure-wrong-tone"
    write_jsonl(production, records)
    out = tmp_path / "estimate.json"

    assert main(
        [*arguments, "--evaluator", str(evaluator_path), "--min-pass-rate", "0.5", "--out", str(out)]
    ) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate_blockers"] == [
        "judged dataset labels were produced by evaluator failure-wrong-tone, "
        "not failure-unsupported-claim"
    ]


def test_estimate_gate_accepts_a_renamed_evaluator_column(tmp_path):
    arguments, validate_arguments, evaluator_path, _ = _validated_judge_fixture(tmp_path)
    assert main(validate_arguments) == 0
    production = tmp_path / "production.jsonl"
    records = [json.loads(line) for line in production.read_text().splitlines()]
    for record in records:
        record["judged_by"] = record.pop("evaluator_id")
    write_jsonl(production, records)
    out = tmp_path / "estimate.json"

    assert main(
        [
            *arguments,
            "--evaluator",
            str(evaluator_path),
            "--dataset-evaluator-column",
            "judged_by",
            "--min-pass-rate",
            "0.8",
            "--out",
            str(out),
        ]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gate_passed"] is True
    assert report["dataset_evaluator_column"] == "judged_by"


def test_validate_judge_fails_ci_when_one_class_is_missed(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {"id": index, "human": label, "judge": "Pass"}
            for label in ("Pass", "Fail")
            for index in range(4)
        ],
    )
    split_dir = tmp_path / "splits"
    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human",
            "--judge-column",
            "judge",
            "--out",
            str(split_dir),
        ]
    ) == 0

    assert main(
        [
            "validate-judge",
            str(split_dir / "test.jsonl"),
            "--human-column",
            "human",
            "--judge-column",
            "judge",
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--bootstrap-iterations",
            "50",
        ]
    ) == 1


def test_validate_judge_rejects_a_different_prediction_field(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {
                "id": index,
                "human": label,
                "judge": label,
                "untrusted_original_field": label,
            }
            for label in ("Pass", "Fail")
            for index in range(4)
        ],
    )
    split_dir = tmp_path / "splits"
    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human",
            "--judge-column",
            "judge",
            "--out",
            str(split_dir),
        ]
    ) == 0

    with pytest.raises(SystemExit):
        main(
            [
                "validate-judge",
                str(split_dir / "test.jsonl"),
                "--human-column",
                "human",
                "--judge-column",
                "untrusted_original_field",
                "--split-manifest",
                str(split_dir / "split_manifest.json"),
            ]
        )


def test_validate_judge_rejects_dataset_that_does_not_match_split_manifest(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {"id": index, "human": label, "judge": label}
            for label in ("Pass", "Fail")
            for index in range(4)
        ],
    )
    split_dir = tmp_path / "splits"
    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human",
            "--judge-column",
            "judge",
            "--out",
            str(split_dir),
        ]
    ) == 0
    test_records = [json.loads(line) for line in (split_dir / "test.jsonl").read_text().splitlines()]
    test_records[0]["human"] = "Fail" if test_records[0]["human"] == "Pass" else "Pass"
    write_jsonl(split_dir / "test.jsonl", test_records)

    with pytest.raises(SystemExit):
        main(
            [
                "validate-judge",
                str(split_dir / "test.jsonl"),
                "--human-column",
                "human",
                "--judge-column",
                "judge",
                "--split-manifest",
                str(split_dir / "split_manifest.json"),
            ]
        )


def test_validate_judge_requires_verified_split_manifest(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [{"human": "Pass", "judge": "Pass"}])

    with pytest.raises(SystemExit):
        main(
            [
                "validate-judge",
                str(labels),
                "--human-column",
                "human",
                "--judge-column",
                "judge",
            ]
        )


def test_validate_judge_requires_all_human_lifecycle_checkpoints(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {
                "id": index,
                "output": f"Agent answer {label} {index} about the refund window",
                "human": label,
                "judge": label,
            }
            for label in ("Pass", "Fail")
            for index in range(4)
        ],
    )
    split_dir = tmp_path / "splits"
    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human",
            "--judge-column",
            "judge",
            "--out",
            str(split_dir),
        ]
    ) == 0
    evaluator_path = tmp_path / "evaluator.json"
    evaluator_path.write_text(
        json.dumps(
            build_evaluator_definition(
                "Unsupported claim",
                {"description": "The output contains an unsupported claim."},
                errors_reviewed=True,
            )
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    assert main(
        [
            "validate-judge",
            str(split_dir / "test.jsonl"),
            "--human-column",
            "human",
            "--judge-column",
            "judge",
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--evaluator",
            str(evaluator_path),
            "--out",
            str(report_path),
            "--bootstrap-iterations",
            "50",
        ]
    ) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    assert report["gate_blockers"] == [
        "human review is incomplete",
        "expert examples do not include both Pass and Fail critiques",
    ]
    assert evaluator["status"] == "validation_failed"
    assert evaluator["validation"]["validated"] is False


def test_validate_judge_enforces_stricter_evaluator_thresholds(tmp_path):
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {"id": f"{label}-{index}", "human": label, "judge": label}
            for label in ("Pass", "Fail")
            for index in range(20)
        ],
    )
    split_dir = tmp_path / "splits"
    assert main(
        [
            "split-labels",
            str(labels),
            "--label-column",
            "human",
            "--judge-column",
            "judge",
            "--out",
            str(split_dir),
        ]
    ) == 0
    test_records = [
        json.loads(line) for line in (split_dir / "test.jsonl").read_text().splitlines()
    ]
    first_pass = next(record for record in test_records if record["human"] == "Pass")
    first_pass["judge"] = "Fail"
    write_jsonl(split_dir / "test.jsonl", test_records)

    evaluator = build_evaluator_definition(
        "Unsupported claim",
        {"description": "The output contains an unsupported claim."},
        source_annotation_ids=["pass", "fail"],
        expert_examples=[
            {"input": {"output": "Grounded"}, "critique": "Grounded answer.", "result": "Pass"},
            {"input": {"output": "Invented"}, "critique": "Unsupported claim.", "result": "Fail"},
        ],
        errors_reviewed=True,
    )
    evaluator["validation"]["minimum_tpr"] = 0.95
    evaluator["validation"]["minimum_tnr"] = 0.90
    evaluator_path = tmp_path / "evaluator.json"
    evaluator_path.write_text(json.dumps(evaluator), encoding="utf-8")
    report_path = tmp_path / "report.json"

    assert main(
        [
            "validate-judge",
            str(split_dir / "test.jsonl"),
            "--human-column",
            "human",
            "--judge-column",
            "judge",
            "--split-manifest",
            str(split_dir / "split_manifest.json"),
            "--evaluator",
            str(evaluator_path),
            "--out",
            str(report_path),
            "--bootstrap-iterations",
            "50",
        ]
    ) == 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    updated_evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    assert report["metrics"]["tpr"] == 0.875
    assert report["thresholds"] == {
        "minimum_tpr": 0.95,
        "minimum_tnr": 0.90,
        "cli_minimum_tpr": 0.8,
        "cli_minimum_tnr": 0.8,
        "evaluator_minimum_tpr": 0.95,
        "evaluator_minimum_tnr": 0.90,
    }
    assert report["gate_passed"] is False
    assert updated_evaluator["status"] == "validation_failed"


if __name__ == "__main__":
    from tests.utils import pytest_this_file

    raise SystemExit(pytest_this_file(__file__))
