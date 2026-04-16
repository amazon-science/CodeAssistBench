# CodeAssistBench

A benchmark for evaluating AI coding assistants on real GitHub issues. Includes a curated dataset of GitHub issues with satisfaction conditions and Dockerfiles for reproducible evaluation.

## 📊 Dataset

We recommend **`dataset/cab_verified_v3.jsonl`** — 274 human-verified issues across 7 languages with **human-reviewed satisfaction conditions**.

| Dataset | Issues | Description |
|---------|--------|-------------|
| **`cab_verified_v3.jsonl`** | **274** | **⭐ Recommended** — Human-verified issues with human-reviewed satisfaction conditions |
| `cab_verified_v2.jsonl` | 274 | Human-verified issues with LLM-generated satisfaction conditions |
| `cab_recent_v2.jsonl` | 771 | Full dataset with LLM-generated satisfaction conditions & classification |
| `cab_recent.jsonl` | 308 | Earlier recent issues (June 2025 – Jan 2026) |
| `cab_verified.jsonl` | 149 | Legacy verified subset with tested Dockerfiles |

**Languages:** Python, JavaScript, TypeScript, Java, Go, C, C++

### Human-Verified Satisfaction Conditions

The satisfaction conditions in `cab_verified_v3.jsonl` were refined through human annotation (raw annotations in `codeassistbench_satisfaction_conditions_2026_03_03_datadelivery_274.json`):

- **92.7%** of entries validated as correct — no changes needed
- **4.7%** had irrelevant conditions removed (18 dropped)
- **3.6%** had missing conditions added (15 added)

### Dataset Fields

```json
{
  "task_id": "cab_verified_1",
  "number": 1234,
  "title": "Bug: Memory leak in parser",
  "url": "https://github.com/owner/repo/issues/1234",
  "body": "When parsing large files...",
  "author": "user123",
  "comments": [{"user": "maintainer", "body": "..."}],
  "satisfaction_conditions": [
    "Memory usage remains stable when parsing files >100MB",
    "No regression in parsing speed for normal files"
  ],
  "commit_id": "abc123...",
  "language": "python"
}
```

---

## ⚡ Quick Start

```bash
# Install
git clone https://github.com/your-org/CodeAssistBench.git
cd CodeAssistBench
pip install -r requirements.txt && pip install -e .

# Set credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2

# Generate maintainer responses
python -m cab_evaluation.cli generation-dataset \
  dataset/cab_verified_v3.jsonl \
  --output results/generation.jsonl \
  --agent-models '{"maintainer": "haiku", "user": "haiku"}' \
  --language python

# Judge the responses
python -m cab_evaluation.cli evaluation-dataset \
  results/generation.jsonl \
  --output results/evaluation.jsonl \
  --agent-models '{"judge": "haiku"}'
```

For production evaluation, use `sonnet` instead of `haiku`.

### Model Aliases (default Strands framework)

| Alias | Model |
|-------|-------|
| `haiku` | Claude 3.5 Haiku |
| `sonnet` | Claude 3.7 Sonnet |
| `sonnet37` | Claude 3.7 Sonnet |

Additional aliases (`sonnet4`, `sonnet45`, `opus`, etc.) are available with the Kiro CLI framework. See [Usage Guide](examples/USAGE_GUIDE.md) for details.

### Verdict Types

| Verdict | Meaning |
|---------|---------|
| `CORRECT` | Fully addresses the issue |
| `PARTIALLY_CORRECT` | Addresses some aspects |
| `INCORRECT` | Wrong or irrelevant |
| `ERROR` | Processing failed |

---

## 📖 Documentation

| Doc | Description |
|-----|-------------|
| **[Usage Guide](examples/USAGE_GUIDE.md)** | Detailed evaluation instructions, output format, analysis examples |
| **[Data Pipeline](docs/DATA_PIPELINE.md)** | How to generate your own dataset from scratch |
| **[Development](DEVELOPMENT.md)** | Contributing and development setup |

---

## 📁 Project Structure

```
CodeAssistBench/
├── dataset/                    # Datasets (JSONL)
├── src/cab_evaluation/         # Evaluation framework
├── script/                     # Data collection & processing scripts
├── prompts/                    # Prompt templates
├── tools/                      # Strands tools for Dockerfile generation
├── examples/                   # Sample data and usage guide
└── docs/                       # Pipeline documentation
```

---

## 📄 Citation

```bibtex
@inproceedings{
kim2025codeassistbench,
title={CodeAssistBench ({CAB}): Dataset \& Benchmarking for Multi-turn Chat-Based Code Assistance},
author={Myeongsoo Kim and Shweta Garg and Baishakhi Ray and Varun Kumar and Anoop Deoras},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=2R6y4Ku9kG}
}
```

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE). GitHub issues are subject to their respective repository licenses.
