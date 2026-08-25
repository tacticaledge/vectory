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

3. If the workspace has an accepted failure mode, create a focused evaluator:

   `vectory promote WORKSPACE "FAILURE MODE" --errors-reviewed`

   Prefer one binary evaluator per failure mode. Treat every promoted evaluator as `draft_unvalidated`.

4. If trusted labels exist, create disjoint sets:

   `vectory split-labels LABELS --label-column HUMAN_LABEL --out SPLITS`

   Use train only for examples or prompt construction, dev for iteration, and test once for final validation.

5. Once evaluator predictions have been added to the held-out test records, run:

   `vectory validate-judge SPLITS/test.jsonl --human-column HUMAN_LABEL --judge-column JUDGE_LABEL --out REPORT`

   A nonzero exit status means the evaluator must not gate releases. Use `--group-by` with product dimensions such as feature, scenario, and persona when those fields exist.

6. For agent reliability and proof-grounded release checks, also run `vectory gate` on Vectory Benchmark traces.

## Preserve these invariants

- Diverse cluster representatives and related-example suggestions are for discovery, never prevalence estimates.
- Inspect complete traces when available, including tool calls, results, and final outputs.
- Keep raw evidence, expert critiques, accepted taxonomy, evaluator definitions, and validation reports inspectable.
- Use expert critiques as judge examples; they must be specific enough for a new teammate to understand the verdict.
- Require both true-pass rate and true-fail rate. Inspect per-class precision and recall, not raw agreement alone.
- Do not claim review, validation, or production readiness that did not happen.
- Re-run expert alignment after material model, prompt, tool, or policy changes.
- Keep sensitive traces local unless the user explicitly authorizes an external model or service.
