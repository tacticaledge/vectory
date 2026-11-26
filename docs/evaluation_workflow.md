# Human-grounded evaluation workflow

Vectory connects failure discovery, domain-expert review, evaluator development, validation, and release gates. The workflow is intentionally artifact-first: every sample choice, critique, failure mode, prompt, and validation result can be inspected as JSON.

## Lifecycle

1. Name the principal domain expert whose judgment defines acceptable behavior.
2. Collect real outputs or traces. Include product dimensions such as feature, scenario, or persona when they matter.
3. Create a diverse discovery sample and review it with binary Pass/Fail verdicts plus detailed critiques.
4. Group failed examples into a human-accepted failure taxonomy and find related traces for confirmation.
5. Fix obvious or pervasive product errors and repeat review before automating the judgment.
6. Promote a remaining failure mode into a focused binary evaluator seeded with expert Pass and Fail critiques from the train split only.
7. Iterate with train and development labels, check for held-out leakage, then validate once on a held-out test set.
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

By default the examples come from the first annotations in the workspace, which may later land in the dev or test split. Once labels are split, draw the examples from the train partition instead:

```bash
vectory promote eval-review "Unsupported claim" \
  --errors-reviewed \
  --split-manifest judge-splits/split_manifest.json \
  --train-split judge-splits/train.jsonl \
  --train-label-column human_label \
  --train-critique-column critique
```

Every train record must carry the binary Pass/Fail label. Records without an expert critique are skipped, because an example without the reasoning behind its verdict cannot teach the judge, and promotion fails when no train record has one. A critiqued record may omit an optional input column, but promotion fails when it carries no content in any requested input field, because an example with an empty input shows the judge nothing to reason about. The critiqued train records must cover both verdicts: promotion fails when they are all Pass or all Fail, since a judge taught one verdict cannot clear both the true-pass and true-fail thresholds. Vectory takes up to four Fail and four Pass records and records their provenance as `train:ID` source annotation IDs.

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
  --judge-column judge_label \
  --out judge-splits
```

Use `train.jsonl` for examples and prompt construction, `dev.jsonl` for iteration, and `test.jsonl` for final validation. Do not move test failures back into prompt development without creating a new untouched test set.

## Check an evaluator for held-out leakage

Duplicated source records, hand-edited prompts, and examples copied from a review screen can all put a held-out record inside an evaluator definition. That inflates the measured accuracy of the judge, the same [data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) pitfall that inflates model scores:

```bash
vectory check-leakage eval-review/evaluators/failure-unsupported-claim.json \
  --split-manifest judge-splits/split_manifest.json \
  --splits dev test \
  --out judge-leakage.json
```

Vectory lowercases each definition section and each record field, collapses whitespace, including the newlines and tabs that serialization escapes, and reports every dev or test record whose text appears in an expert example input, an expert critique, a failure-mode quote, a Pass/Fail definition, a rule pattern, or the prompt. Each record text is searched both verbatim and in its JSON-escaped form, so a record containing quotes or backslashes is still found inside a serialized example or a pasted prompt fragment. The report names the split, record index, record `id`, field, and definition location, and the command exits nonzero when anything leaked.

Only held-out splits can be checked: `--splits` accepts `dev` and `test` and defaults to both. Train examples belong in the definition, so a clean train result would prove nothing about the records the judge is measured on.

Each split file is verified against the manifest before anything is compared: the record count and the independent fingerprint of every immutable field must still match the partition written by `split-labels`, while the declared judge prediction column stays writable. A replaced, edited, or truncated held-out file therefore fails the command instead of producing a clean report for data that is no longer the original partition, which is also why the check reads the files the manifest names rather than paths supplied on the command line.

By default Vectory compares the evaluator's own `input_fields`; use `--text-fields` when the split uses other column names. Texts shorter than `--min-characters` (default 8) are skipped, because short phrases such as a bare verdict match instruction wording without indicating leakage. The report counts those skipped texts, lists every compared field that no record carried in `fields_without_records`, and records how many texts each field actually contributed in `compared_texts`. A check that compared nothing is never reported as clean: when no held-out text reached the comparison, because none of the compared fields exists in the records or because every value was shorter than `--min-characters`, the report sets `clean` to false and the command exits nonzero, so a mismatched field name or a verdict-only column cannot look like a passed check.

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

When `--evaluator` is supplied, Vectory runs the same leakage check against the test records and embeds the result in the report. An evaluator whose definition contains test text is refused: the gate fails with a blocker naming the leaked records, because its measured accuracy would include records it was built from. The gate also fails when the check could not compare anything, that is when the test records carry none of the compared fields or every value is too short to compare; point `--text-fields` at columns the test split actually has, and use `--min-leakage-characters` to control the shortest text compared.

The split manifest fingerprints every original test field independently and designates the judge prediction column at split time. You may add or replace that one declared output column after splitting; Vectory verifies that labels, inputs, metadata, ordering, and record count still match the untouched test partition, and rejects a validator that names a different mutable field. The report includes this provenance, the confusion matrix, balanced accuracy, per-class precision/recall/F1, bootstrap intervals, product-dimension slices, thresholds, and gate result. When an evaluator artifact declares stricter thresholds than the CLI baseline, Vectory enforces the stricter values and records both threshold sources in the report. Updating an evaluator requires verified split provenance and appends the report to its validation history. The command exits nonzero when provenance fails, the gate fails, either class is absent, the evaluator's definition overlaps the test split, the leakage check compared no field, or the evaluator's fix-review checkpoint is incomplete.

Raw agreement alone can be misleading on imbalanced data. Inspect false passes in particular: these are human failures that the automated judge allowed through.

## Estimate the true pass rate

A judge's raw pass rate on new outputs is biased by its own errors. Once a judge is validated, `vectory estimate` corrects the observed rate with the Rogan-Gladen estimator, `(observed + TNR - 1) / (TPR + TNR - 1)`, using the TPR and TNR measured on the held-out test set:

```bash
vectory estimate production-judged.jsonl \
  --judge-column judge_label \
  --test-set judge-splits/test.jsonl \
  --human-column human_label \
  --split-manifest judge-splits/split_manifest.json \
  --evaluator eval-review/evaluators/failure-unsupported-claim.json \
  --min-pass-rate 0.90 \
  --out success-estimate.json
```

The report includes the observed rate, the corrected rate, and a 95% bootstrap interval. The interval resamples the test pairs and the judged outputs together, 20,000 times by default, so it reflects uncertainty in both the judge's accuracy and the observed rate. A resample that lacks a class or leaves the judge at or below chance cannot be corrected, so it counts as 0 for the lower bound and 1 for the upper bound; a weak judge therefore produces a wide interval rather than a falsely narrow one. The report counts these resamples.

`--min-pass-rate` requires `--evaluator`. The command exits nonzero unless the evaluator is validated, its latest validation report fingerprints the same ordered human and judge labels as the test set, every judged row attributes its label to that same evaluator, and the interval's lower bound reaches the threshold. Attribution comes from an `evaluator_id` field on each judged record, renameable with `--dataset-evaluator-column`: a row with no evaluator, or one naming a different evaluator, blocks the gate, because a validated judge's measured accuracy says nothing about labels another judge produced. Correction is refused when TPR + TNR is 1 or less, because such a judge carries no signal. Raising TPR usually narrows the interval the most.

Use a random sample of outputs for the estimate. Coverage-biased discovery samples do not represent production prevalence.

## Score a checklist rubric

A single Pass/Fail verdict says that a response was unacceptable, not which requirement it missed. When one failure mode covers several independent requirements, decompose it into a checklist of atomic yes/no criteria and score each one separately, as [HealthBench](https://openai.com/index/healthbench/) does with physician-written rubrics and as [checklist feedback](https://arxiv.org/abs/2410.03608) does for instruction following.

A rubric spec is a small JSON artifact:

```json
{
  "schema_version": "1.0",
  "rubric_id": "refund-reply",
  "criteria": [
    {"id": "states_window", "text": "States the refund window that applies.", "weight": 3},
    {"id": "cites_policy", "text": "Cites the policy section the answer relies on.", "weight": 2},
    {"id": "next_step", "text": "Tells the customer the next concrete step.", "weight": 1},
    {"id": "invents_terms", "text": "Invents a refund term.", "weight": 2, "polarity": "negative"}
  ]
}
```

The spec is validated before any judge call: ids must be unique, every weight must be a plain finite number rather than a string or boolean, and at least one positive criterion must carry a weight above zero, because an all-zero weight set leaves the score with no denominator. A criterion with `"polarity": "negative"` is a penalty: meeting it subtracts its weight from the numerator and it never joins the denominator.

The judge returns a critique first and then one verdict per id. Vectory computes the rollup deterministically as the clipped weighted fraction `sum(w * met) / sum(w)` over the positive criteria the judge resolved. Criteria the judge skipped, contradicted with two different verdicts, or returned unparseably are reported as `unknown`: they stay out of both sums instead of counting as met, so an incomplete response lowers coverage rather than the score. A response that is not usable JSON at all leaves every criterion `unknown` and is reported as unscored.

Score recorded judge responses from the CLI:

```bash
vectory score-rubric data/rubric/example_judge_responses.jsonl \
  --rubric data/rubric/example_rubric.json \
  --response-column judge_response \
  --out rubric-report.json \
  --scores-out rubric-scores.jsonl
```

The report ranks criteria by dataset-level failure rate, the share of the responses that resolved a criterion and failed it: not met for a positive criterion, met for a penalty, because a met penalty is the defect it names. Equal rates are ordered by weight, and a criterion no response resolved has no rate and ranks last with its `unknown` count. The report is derived only from the supplied responses and carries no timestamps, so re-running it over the same recorded outputs produces byte-identical JSON and a reviewable diff. `--scores-out` writes one row per response with its score and per-criterion status.

In the library, `components.evaluators.rubric` exposes the spec validator, the prompt checklist renderer, the response parser, and the aggregator, and `LLMJudgeEvaluator(decision_mode="rubric", rubric=...)` runs the same prompt and parser against a provider. Each scored response records the `rubric_id` it was judged against, and the aggregator rejects any result that names a different rubric, because two rubrics can reuse the same criterion ids while weighting or inverting them differently.

Rubric scores are diagnostics, not release gates: they are weighted judge opinions, not human-validated decisions. Keep gating on a binary judge whose true-pass and true-fail rates were measured on held-out human labels, and use the criterion ranking to decide which requirement to fix or which failure mode to promote next.

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

## Public dependency boundary

Vectory is a public MIT-licensed package. Tactical Edge's shared `commons-python` repository is private, so it cannot be a runtime or test dependency of this project. The packaged `components.agentic.Bot`, typed provider settings, and `tests.utils` helper are intentional public-repository adapters for the small interfaces Vectory needs. Keep these adapters narrow; move them to a shared package only when that package is publicly installable and versioned for external consumers.

## Design sources

The workflow adapts the error-discovery loop from [AI Evals for Engineers](https://github.com/ai-evals-course/evals-skills) and the critique-shadowing process in [Using LLM-as-a-Judge for Evaluation](https://hamel.dev/blog/posts/llm-judge/). Vectory's implementation is independent and uses its own versioned artifacts, UI, CLI, validation metrics, and Vectory Benchmark gates.
