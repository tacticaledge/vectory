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
    assert cli_version == component_version == "1.1.1"
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
