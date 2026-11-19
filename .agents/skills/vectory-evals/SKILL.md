---
name: vectory-evals
description: "Run Vectory's local AI evaluation lifecycle on output or trace exports: diverse failure discovery, human review workspaces, failure-mode promotion, label splitting, binary judge validation, and CI gates. Use when asked to inspect AI traces, find failure modes, create an evaluator from reviewed errors, audit an eval pipeline, or validate an AI judge."
---

# Vectory evaluation workflow

Use Vectory's deterministic CLI and inspectable JSON artifacts. Start by inspecting `vectory --help`; in a source checkout, use `python3 -m vectory_cli.cli` if the console command is unavailable.

## Route by current evidence

1. If the user has outputs or agent traces but no taxonomy, run:

   `vectory discover INPUT --workspace WORKSPACE`

   Ask the principal domain expert to review `samples.json` in the Vectory app, record binary verdicts and detailed critiques in `annotations.json`, and accept a taxonomy in `taxonomy.json`. Never invent or silently substitute human annotations.

2. Fix obvious product errors revealed by review. Repeat the review until pervasive errors stabilize.

3. If trusted labels exist, create disjoint sets before building judge examples:

   `vectory split-labels LABELS --label-column HUMAN_LABEL --judge-column JUDGE_LABEL --out SPLITS`

   Use train only for examples or prompt construction, dev for iteration, and test once for final validation.

4. If the workspace has an accepted failure mode, create a focused evaluator:

   `vectory promote WORKSPACE "FAILURE MODE" --errors-reviewed --train-split SPLITS/train.jsonl`

   Drawing examples from the train split keeps dev and test records out of the definition. Every train record needs the Pass/Fail label, only records carrying an expert critique become examples, and those critiqued records must cover both verdicts. Without `--train-split`, examples come from the first workspace annotations, which may later land in dev or test. Prefer one binary evaluator per failure mode. Treat every promoted evaluator as `draft_unvalidated`.

5. Before validating, confirm no held-out record reached the definition:

   `vectory check-leakage EVALUATOR_JSON --split-manifest SPLITS/split_manifest.json`

   A nonzero exit status lists every dev or test record whose normalized text appears in the prompt, examples, definitions, or rule. Remove or replace those examples; never move the leaked records into another split. The check also fails when no held-out text was compared at all, because it then proved nothing; name the real columns with `--text-fields` instead of verdict-only columns.

6. Once evaluator predictions have been added to the held-out test records, run:

   `vectory validate-judge SPLITS/test.jsonl --human-column HUMAN_LABEL --judge-column JUDGE_LABEL --split-manifest SPLITS/split_manifest.json --out REPORT`

   A nonzero exit status means the evaluator must not gate releases, including when `--evaluator` is supplied and its definition overlaps the test split. Use `--group-by` with product dimensions such as feature, scenario, and persona when those fields exist.

7. To report how often a validated evaluator's target actually passes on a random sample of new outputs, run:

   `vectory estimate JUDGED --judge-column JUDGE_LABEL --test-set SPLITS/test.jsonl --human-column HUMAN_LABEL --split-manifest SPLITS/split_manifest.json --out ESTIMATE`

   Report the corrected rate with its 95% interval, not the raw judge pass rate. To gate on the interval's lower bound, add `--min-pass-rate` together with `--evaluator EVALUATOR_JSON`; only a validated evaluator can gate, and every judged row must name that evaluator in its `evaluator_id` field.

8. For agent reliability and proof-grounded release checks, also run `vectory gate` on Vectory Benchmark traces.

9. When reliability under repetition matters, submit several runs per `task_id` to `vectory benchmark`. The leaderboard and report bundle then add unbiased `pass@k` and `pass^k`; quote `pass^k`, success on every one of `k` tries, for production readiness, and say which `k` and how many repeated tasks it covers.

## Preserve these invariants

- Diverse cluster representatives and related-example suggestions are for discovery, never prevalence estimates.
- Inspect complete traces when available, including tool calls, results, and final outputs.
- Keep raw evidence, expert critiques, accepted taxonomy, evaluator definitions, and validation reports inspectable.
- Use expert critiques as judge examples; they must be specific enough for a new teammate to understand the verdict.
- Draw judge examples from the train split only, and never from dev or test records.
- Require both true-pass rate and true-fail rate. Inspect per-class precision and recall, not raw agreement alone.
- Do not claim review, validation, or production readiness that did not happen.
- Re-run expert alignment after material model, prompt, tool, or policy changes.
- Keep sensitive traces local unless the user explicitly authorizes an external model or service.
