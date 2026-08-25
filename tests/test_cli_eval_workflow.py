import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from vectory_cli.cli import main


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


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
            "scenario": "common" if index % 4 < 2 else "edge",
        }
        for index in range(40)
    ]
    write_jsonl(labels, records)
    split_dir = tmp_path / "splits"

    assert main(
        ["split-labels", str(labels), "--label-column", "human_label", "--out", str(split_dir)]
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


def test_validate_judge_fails_ci_when_one_class_is_missed(tmp_path):
    labels = tmp_path / "test.jsonl"
    write_jsonl(
        labels,
        [
            {"human": "Pass", "judge": "Pass"},
            {"human": "Pass", "judge": "Pass"},
            {"human": "Fail", "judge": "Pass"},
            {"human": "Fail", "judge": "Pass"},
        ],
    )

    assert main(
        [
            "validate-judge",
            str(labels),
            "--human-column",
            "human",
            "--judge-column",
            "judge",
            "--bootstrap-iterations",
            "50",
        ]
    ) == 1


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
        ["split-labels", str(labels), "--label-column", "human", "--out", str(split_dir)]
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
