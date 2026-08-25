# Human-grounded evaluation workflow

Vectory connects failure discovery, domain-expert review, evaluator development, validation, and release gates. The workflow is intentionally artifact-first: every sample choice, critique, failure mode, prompt, and validation result can be inspected as JSON.

## Lifecycle

1. Name the principal domain expert whose judgment defines acceptable behavior.
2. Collect real outputs or traces. Include product dimensions such as feature, scenario, or persona when they matter.
3. Create a diverse discovery sample and review it with binary Pass/Fail verdicts plus detailed critiques.
4. Group failed examples into a human-accepted failure taxonomy and find related traces for confirmation.
5. Fix obvious or pervasive product errors and repeat review before automating the judgment.
6. Promote a remaining failure mode into a focused binary evaluator seeded with expert Pass and Fail critiques.
7. Iterate with train and development labels, then validate once on a held-out test set.
8. Gate only when both true-pass and true-fail rates clear the agreed thresholds. Repeat after material changes.

The default discovery sample is 30 records. It combines product-dimension coverage, cluster representatives, and random exploration. This is a coverage-biased sample and must not be used to estimate production prevalence.

## Start in the app

Run `vectory app`, then open **🚀 Start**. It routes to Dataset, Error Analysis, evaluator validation, an eval-pipeline audit, or Vectory Benchmark based on the current evidence.

On **🔍 Error Analysis**:

- enter the principal reviewer and optional coverage dimensions;
- review ordered trace segments on one screen;
- record a Pass/Fail verdict and required expert critique;
- build and accept failure modes;
- retrieve related candidates and add them to the review queue;
- review failures by product dimension;
- promote accepted modes into draft binary evaluators; and
- export a portable review bundle.

The **🤖 LLM Judge** page is binary by default and can consume a promoted evaluator definition. A 1–5 exploratory mode remains available for compatibility, but it is not suitable for release gates without independent calibration.

## Create a review workspace from the CLI

```bash
vectory discover traces.jsonl \
  --workspace eval-review \
  --sample-size 30 \
  --dimension-fields feature scenario persona \
  --reviewer "domain-expert-id" \
  --reviewer-role "Support Director"
```

If the workspace already exists, `--force` archives the complete directory to a timestamped sibling before creating a fresh workspace. Human annotations, accepted taxonomy, suggestions, and promoted evaluators are never overwritten in place.

The workspace contains:

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Schema, source hash, reviewer protocol, dimensions, and sampling warning |
| `records.jsonl` | Immutable review source |
| `samples.json` | Selected indices, strategy, cluster, and selection rationale |
| `annotations.json` | Expert verdicts and detailed critiques |
| `taxonomy.json` | Accepted failure modes |
| `suggestions.json` | Machine suggestions awaiting human confirmation |
| `evaluators/*.json` | Promoted evaluator definitions and validation history |

The app can export the same information as one self-contained `vectory_review_bundle.json`.

## Promote a failure mode

After populating annotations and accepting a taxonomy entry:

```bash
vectory promote eval-review "Unsupported claim" --errors-reviewed
```

The generated evaluator:

- evaluates exactly one failure mode;
- returns critique-first binary JSON;
- includes up to four expert Fail and four expert Pass examples;
- records its source annotation IDs and lifecycle checkpoints; and
- starts as `draft_unvalidated`.

Regex and substring rules are available when code is simpler and more reliable:

```bash
vectory promote eval-review "Secret leakage" \
  --kind regex \
  --pattern 'sk-[A-Za-z0-9]{20,}' \
  --errors-reviewed
```

## Split labels without leakage

Trusted labels must contain at least three Pass and three Fail examples:

```bash
vectory split-labels judge-labels.jsonl \
  --label-column human_label \
  --out judge-splits
```

Use `train.jsonl` for examples and prompt construction, `dev.jsonl` for iteration, and `test.jsonl` for final validation. Do not move test failures back into prompt development without creating a new untouched test set.

## Validate and gate a judge

After adding evaluator predictions to the held-out test records:

```bash
vectory validate-judge judge-splits/test.jsonl \
  --human-column human_label \
  --judge-column judge_label \
  --split-manifest judge-splits/split_manifest.json \
  --min-tpr 0.80 \
  --min-tnr 0.80 \
  --group-by feature scenario persona \
  --out judge-validation.json \
  --evaluator eval-review/evaluators/failure-unsupported-claim.json
```

The split manifest fingerprints every original test field independently. You may add or replace the declared judge prediction column after splitting; Vectory excludes only that output column while verifying that labels, inputs, metadata, ordering, and record count still match the untouched test partition. The report includes this provenance, the confusion matrix, balanced accuracy, per-class precision/recall/F1, bootstrap intervals, product-dimension slices, thresholds, and gate result. Updating an evaluator requires verified split provenance and appends the report to its validation history. The command exits nonzero when provenance fails, the gate fails, either class is absent, or the evaluator's fix-review checkpoint is incomplete.

Raw agreement alone can be misleading on imbalanced data. Inspect false passes in particular: these are human failures that the automated judge allowed through.

## Use with Vectory Benchmark

The human-grounded workflow captures product-specific expectations. Vectory Benchmark supplies complementary deterministic and proof-grounded agent checks:

```bash
vectory gate submission.jsonl \
  --min-score 0.90 \
  --block-severity critical \
  --report-out vectory-report
```

Use both when releasing agents: validated binary judges cover learned domain expectations, while Vectory Benchmark checks trace quality, tool discipline, control, recovery, and proof grounding.

## Agent workflow

The repository includes `.agents/skills/vectory-evals`. Codex can discover this project skill automatically from a source checkout and use it to route trace-review, evaluator-promotion, validation, and audit requests. The skill preserves the same human-review and held-out-validation constraints as the app and CLI.

## Design sources

The workflow adapts the error-discovery loop from [AI Evals for Engineers](https://github.com/ai-evals-course/evals-skills) and the critique-shadowing process in [Using LLM-as-a-Judge for Evaluation](https://hamel.dev/blog/posts/llm-judge/). Vectory's implementation is independent and uses its own versioned artifacts, UI, CLI, validation metrics, and Vectory Benchmark gates.
