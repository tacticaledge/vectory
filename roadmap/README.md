# Automated roadmap

`queue.json` lists features the automation pipeline builds. Items ship in priority order (1 first), then oldest first.

| Workflow | Schedule (UTC) | What it does |
| --- | --- | --- |
| `oss-research.yml` | Monday 13:00 | Researches recent evaluation methods and tools with web access only, then opens a pull request that adds up to three new items to this queue. |
| `oss-build.yml` | Monday and Thursday 14:00 | Implements the next queued item with tests and docs, then opens a pull request. Skips the run while an earlier feature pull request is open. |
| `oss-review-loop.yml` | On each Connectory review | Merges the exact reviewed commit once Connectory reports it ready to merge and tests pass on it. Otherwise applies an automated fix round. After five reviews without a clean result it labels the pull request `needs-human` and stops. |
| `oss-release.yml` | Friday 15:00 | When features or fixes merged since the last tag, bumps the version, updates `CHANGELOG.md`, tags, creates a GitHub release, and runs the PyPI publish workflow. |

## Safeguards

- Web research runs in a job without repository write access, and its output must pass a strict schema before it can enter the queue.
- Agents run with model-only cloud credentials and no GitHub token. A separate job pushes their changes.
- `.github/automation/guard_diff.py` rejects automated changes to CI, packaging, dependencies, release metadata, and infrastructure. It also rejects binary files, diffs over 1,500 lines, and source changes without tests. The merge step always runs the copy of the guard on `main`.
- Only pull requests opened by the workflow from `automation/` branches in this repository are merged automatically.

## Maintainer controls

- Add, reorder, or block items by editing `queue.json`. Set `status` to `blocked` to skip an item.
- Build a specific item: run `oss-build.yml` manually with `item_id`.
- Pause the pipeline: disable the workflows in the Actions tab, or leave a `needs-human` pull request open.
- Requires the `BEDROCK_AUTOMATION_ROLE_ARN` secret, an AWS role trusted for this repository through GitHub OIDC with Bedrock model access. Set the `BEDROCK_CLAUDE_MODEL` variable to override the model.
