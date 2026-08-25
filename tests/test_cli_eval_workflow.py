import json

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


def test_split_and_validate_judge_gate(tmp_path):
    labels = tmp_path / "labels.jsonl"
    records = [
        {
            "id": index,
            "human_label": "Pass" if index % 2 == 0 else "Fail",
            "judge_label": "Pass" if index % 2 == 0 else "Fail",
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

    report = tmp_path / "validation.json"
    assert main(
        [
            "validate-judge",
            str(labels),
            "--human-column",
            "human_label",
            "--judge-column",
            "judge_label",
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
