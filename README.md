# Vectory

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Precision LLM and agent evaluation** - An open-source toolkit for custom benchmarks, AI judges, deterministic metrics, RAG scoring, human review, and proof-grounded trace validation.

## Features

- **📊 Dataset Management** - Upload JSON, CSV, or PDF files. Automatic column detection and mapping.
- **🤖 LLM-as-Judge** - Use current OpenAI or Anthropic models to automatically evaluate output quality with customizable criteria.
- **📏 Rule-Based Metrics** - Calculate BLEU, ROUGE, exact match, Levenshtein similarity, and regex patterns.
- **🔍 Error Analysis** - Discover failure modes with open coding, axial coding, taxonomy dashboards, and AI-assisted taxonomy suggestions.
- **🔎 RAG Metrics** - Evaluate retrieval precision, recall, MRR, NDCG, context relevance, faithfulness, and answer completeness.
- **👤 Human Evaluation** - Rate outputs manually with customizable scales and criteria.
- **🏆 MTEB Leaderboard** - View embedding model benchmarks from the Massive Text Embedding Benchmark.
- **🧪 Custom Benchmarks** - Prepare organization datasets and run optional local embedding model comparisons.
- **🧭 Vectory Benchmark** - Score submitted agent traces for task success, reality sampling, tool discipline, recovery, control, turn efficiency, and proof grounding.
- **⚙️ Configurable Themes** - Switch between Dark, Light, and Colorblind-safe themes for accessibility.

## Quick Start

```bash
pip install vectoryai
vectory app
```

Use `pipx` when you want an isolated CLI install:

```bash
pipx install vectoryai
vectory app
```

The app opens in your browser at `http://localhost:8501`.

Run the included agent benchmark example from the CLI:

```bash
vectory benchmark data/vectory_benchmark/example_submission.json
```

Run the included proof-grounding example:

```bash
vectory benchmark data/vectory_benchmark/example_proof_submission.json
```

Run the policy proof validation example:

```bash
vectory benchmark data/vectory_benchmark/example_policy_proof_submission.json
```

Run the matching policy proof failure example:

```bash
vectory benchmark data/vectory_benchmark/example_policy_proof_failure_submission.json
```

Write an inspectable benchmark report bundle:

```bash
vectory benchmark data/vectory_benchmark/example_policy_proof_submission.json \
  --report-out /tmp/vectory-report
```

Use Vectory as a CI release gate:

```bash
vectory gate data/vectory_benchmark/example_policy_proof_submission.json \
  --min-score 0.90 \
  --block-severity critical \
  --report-out /tmp/vectory-gate-report
```

## Installation

### Local CLI

Use `pip` inside an activated virtual environment or project environment:

```bash
pip install vectoryai
vectory app
```

Use `pipx` when you want Vectory as a standalone local command without mixing its dependencies into another Python project:

```bash
pipx install vectoryai
vectory app --port 8501
```

### Source Checkout

```bash
git clone https://github.com/tacticaledge/vectory.git
cd vectory
pip install -e ".[dev]"
vectory app
```

### Prerequisites

- Python 3.10 or higher
- pip, pipx, or uv

### Contributor Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the project in editable mode
pip install -e ".[dev]"

# Launch the app
vectory app
```

### Optional: Formal Checker Support

The default install does not execute local formal checkers. Proof artifacts submitted in traces are still scored deterministically. Install the optional formal extra only when you want Vectory to run trusted suite-defined checker commands locally:

```bash
pip install "vectoryai[formal]"
```

Formal runtime is disabled by default. Enable it explicitly and provide a local workspace:

```bash
vectory benchmark path/to/submission.json \
  --suite path/to/manifest.json \
  --workspace path/to/workspace \
  --allow-formal-runtime
```

Uploaded submissions cannot provide executable commands. Vectory only runs checker commands defined in the trusted local suite manifest.

### Optional: Embedding Benchmark Support

The default install does not include PyTorch. Embedding comparison features are optional because they add large model dependencies. Install them only when you need local embedding benchmark runs:

```bash
pip install "vectoryai[embedding]"
```

For a source checkout, install the optional extra from the checkout:

```bash
pip install -e ".[embedding]"
```

For an existing `pipx` install, add the optional runtime to the isolated Vectory environment:

```bash
pipx inject vectoryai "torch>=2.1.0" mteb sentence-transformers
```

## Usage

### 1. Upload Your Data

Navigate to the **📊 Dataset** page and upload your evaluation data:

- **JSON/JSONL**: Arrays of objects with input/output/expected fields
- **CSV**: Tabular data with headers
- **PDF**: Extracted to markdown (optimized for LLM consumption)

The app will automatically detect column roles, or you can map them manually.

### 2. Run Evaluations

Choose your evaluation method:

#### LLM-as-Judge (🤖)
- Select provider (OpenAI or Anthropic)
- Enter your API key
- Choose evaluation criteria
- Run batch evaluation

#### Rule-Based Metrics (📏)
- Select metrics (BLEU, ROUGE, exact match, etc.)
- Run evaluation
- View per-sample and aggregate results

#### Error Analysis (🔍)
- Review traces using open coding and pass/fail decisions
- Group notes into a structured failure-mode taxonomy
- Generate AI-assisted taxonomy suggestions from reviewer notes
- Export annotations, taxonomies, and summary reports

#### Human Evaluation (👤)
- Navigate through samples
- Rate on customizable scales
- Add feedback and criteria tags
- Export annotations

#### Vectory Benchmark (🧭)
- Upload agent run traces as JSON or JSONL
- Score submitted runs against a built-in agent reliability manifest
- Review pathologies such as premature planning, retrieval thrashing, ignored evidence, approval bypass, open proof obligations, failed checkers, and placeholder proofs
- Test policy proof traces where final claims must connect to policy rules, source references, proof obligations, and checker results
- Export run scores, leaderboard summaries, benchmark cards, claim-evidence tables, checkpoint timelines, and static HTML report bundles
- Use CI gates to block releases when scores fall below threshold or critical pathologies appear

Vectory Benchmark is designed for teams that need to understand agent behavior from real traces: whether the agent observed before acting, used tools productively, recovered from failure, respected scope, and stopped before extra turns became low-value work. For private evaluations of internal agents and workflows, use the Vectory contact form: https://vectoryai.com/#contact

Run the included example from the command line:

```bash
vectory benchmark data/vectory_benchmark/example_submission.json
```

Run a proof-grounded policy sample:

```bash
vectory benchmark data/vectory_benchmark/example_policy_proof_submission.json
```

This sample shows why proof grounding is different from ordinary LLM-as-judge, keyword checks, or RAG metrics. The final answer only receives full credit when `claims` link to `evidence`, required `proof_obligations` are closed, policy `checker_results` pass, and `NO_TRANSLATIONS` warnings are preserved instead of silently treated as proved facts.

Use the failure fixture to see the expected negative behavior:

```bash
vectory benchmark data/vectory_benchmark/example_policy_proof_failure_submission.json
```

The bad trace claims success despite failed checkers, open obligations, and unsupported evidence, so Vectory should flag pathologies such as `unclosed_proof_obligation`, `ignored_checker_result`, `policy_regression`, and `evidence_does_not_support_claim`.

Export scores and leaderboard summaries:

```bash
vectory benchmark data/vectory_benchmark/example_submission.json \
  --scores-out vectory-scores.json \
  --leaderboard-out vectory-leaderboard.csv
```

Export a static report bundle:

```bash
vectory benchmark data/vectory_benchmark/example_policy_proof_submission.json \
  --report-out /tmp/vectory-report
```

The report bundle includes `index.html`, `scores.json`, `leaderboard.json`, `pathologies.json`, `claim_evidence_table.json`, `checkpoints.json`, and `benchmark_card.json`.

Run a release gate:

```bash
vectory gate data/vectory_benchmark/example_policy_proof_submission.json \
  --min-score 0.90 \
  --block-severity critical
```

The gate exits non-zero when a run falls below the threshold, fails its task gate, exceeds the optional pathology-risk limit, or includes a pathology at or above the configured severity.

### 3. Export Results

Download your evaluation results as CSV or JSON from any results page.

## Configuration

### API Keys

Configure API keys in one of three ways:

#### Option 1: Streamlit Secrets (Recommended for deployment)

Create `.streamlit/secrets.toml`:

```toml
[api_keys]
openai = "<openai-api-key>"
anthropic = "<anthropic-api-key>"
```

#### Option 2: Environment Variables

```bash
export OPENAI_API_KEY="<openai-api-key>"
export ANTHROPIC_API_KEY="<anthropic-api-key>"
```

#### Option 3: Manual Input

Enter your API key directly in the UI (stored only in session).

### Themes & Accessibility

Vectory includes multiple themes to suit different preferences and accessibility needs. Change your theme in the **⚙️ Settings** page.

#### Available Themes:

| Theme | Description |
|-------|-------------|
| **Dark Developer** | Modern dark theme optimized for developers (default) |
| **Light** | Clean light theme for bright environments |
| **High Contrast (Colorblind Safe)** | Blue-orange palette distinguishable by most forms of color blindness |
| **High Contrast Dark** | Maximum contrast for users with low vision |

The colorblind-safe theme uses a carefully selected palette that works for:
- Deuteranopia (red-green color blindness)
- Protanopia (red color blindness)
- Tritanopia (blue-yellow color blindness)

### Streamlit Configuration

The default theme is configured in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#f1f5f9"
font = "monospace"
```

## Deployment

### Python Package Release

Vectory is packaged for Python distribution as `vectoryai` and exposes the `vectory` command. Release publishing to PyPI is handled by the `Publish Python Package` GitHub Actions workflow when a version tag such as `v1.0.0` is pushed, or when the workflow is run manually.

Publishing uses PyPI Trusted Publishing. Internal package mirrors, when enabled, use GitHub Actions secrets and variables that are not committed to the public repository.

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install .

EXPOSE 8501

CMD ["vectory", "app", "--address", "0.0.0.0"]
```

### Manual Deployment

```bash
vectory app --port 8501 --address 0.0.0.0
```

## Project Structure

```
vectory/
├── app.py                    # Main entry point
├── requirements.txt          # Python dependencies
├── pages/                    # Streamlit pages
│   ├── 1_📊_Dataset.py      # Dataset upload & config
│   ├── 2_🤖_LLM_Judge.py    # LLM-as-Judge evaluation
│   ├── 3_📏_Metrics.py      # Rule-based metrics
│   ├── 4_👤_Human_Eval.py   # Human evaluation
│   ├── 5_🏆_Leaderboard.py  # MTEB leaderboard
│   ├── 6_🧪_Custom_Benchmark.py
│   ├── 7_⚙️_Settings.py     # Theme & API key settings
│   ├── 8_🔍_Error_Analysis.py
│   └── 9_🧭_VectoryBenchmark.py
├── components/               # Core modules
│   ├── __init__.py          # Version and branding
│   ├── models.py            # Pydantic data models
│   ├── themes.py            # Configurable theme system
│   ├── document_loader.py   # File loading utilities
│   ├── error_discovery.py   # AI-assisted taxonomy helpers
│   ├── ui.py                # Animated UI components
│   ├── mteb_evaluator.py    # Embedding evaluation
│   └── evaluators/          # Evaluation implementations
│       ├── llm_judge.py
│       ├── rag_metrics.py
│       └── metrics.py
├── tests/                    # Test suite
└── data/                     # Sample datasets
```

## Testing

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=components --cov-report=html
```

## Troubleshooting

### PyTorch Version Error

If you see `module 'torch' has no attribute 'compiler'`:

```bash
pip install -e ".[embedding]"
```

The app works without PyTorch. Local embedding model execution stays disabled until the optional embedding extra is installed.

### NLTK Data Missing

If you see NLTK-related errors:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### API Key Issues

- Ensure your API key is valid and has sufficient credits
- Check that the key is correctly formatted (no extra spaces)
- For Anthropic, ensure you're using the correct model names

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) - The amazing framework powering this app
- [MTEB](https://github.com/embeddings-benchmark/mteb) - Massive Text Embedding Benchmark
- [PyMuPDF4LLM](https://pymupdf.readthedocs.io/) - PDF to Markdown conversion
- [HuggingFace Evaluate](https://huggingface.co/docs/evaluate/) - Evaluation metrics

---

**Vectory** - Precision LLM Evaluation | [Tactical Edge](https://tacticaledgeai.com)
