# CodeAssistBench

A benchmark for evaluating AI coding assistants on real GitHub issues. This project includes a curated dataset of GitHub issues with Dockerfiles for reproducible evaluation, plus tools for dataset creation and AI agent evaluation.

## 📊 Dataset Overview

CodeAssistBench provides two ready-to-use datasets:

| Dataset | Issues | Languages | Description |
|---------|--------|-----------|-------------|
| `dataset/cab_recent.jsonl` | 308 | 7 | Recent issues (June 2025 - Jan 2026) |
| `dataset/cab_verified.jsonl` | 149 | 7 | Verified subset with tested Dockerfiles |

### Language Distribution

| Language | Issues | With Dockerfiles |
|----------|--------|------------------|
| Python | ~50 | ✓ |
| JavaScript | ~45 | ✓ |
| TypeScript | ~50 | ✓ |
| Java | ~40 | ✓ |
| Go | ~45 | ✓ |
| C | ~40 | ✓ |
| C++ | ~38 | ✓ |

### Dataset Fields

Each issue in the dataset contains:

```json
{
  "number": 1234,
  "title": "Bug: Memory leak in parser",
  "created_at": "2025-07-15T10:30:00Z",
  "closed_at": "2025-07-20T14:22:00Z",
  "commit_id": "abc123def456...",
  "labels": ["bug", "parser"],
  "url": "https://github.com/owner/repo/issues/1234",
  "body": "When parsing large files, memory usage grows unbounded...",
  "author": "user123",
  "comments": [
    {
      "user": "maintainer",
      "created_at": "2025-07-16T08:00:00Z",
      "body": "Thanks for reporting! Can you share the file?"
    }
  ],
  "satisfaction_conditions": [
    "Memory usage remains stable when parsing files >100MB",
    "Parser handles all edge cases mentioned in the issue",
    "No regression in parsing speed for normal files"
  ],
  "_classification": {
    "category": "Can be dockerized without any issue",
    "timestamp": "2025-04-14 01:01:54"
  },
  "dockerfile": "FROM python:3.11-slim\n...",
  "language": "python"
}
```

## 🚀 Quick Start

### Using the Dataset

```python
import json

# Load the dataset
with open('dataset/cab_recent.jsonl', 'r') as f:
    issues = [json.loads(line) for line in f]

# Filter by language
python_issues = [i for i in issues if i.get('language') == 'python']

# Get issues with Dockerfiles
dockerized = [i for i in issues if i.get('dockerfile')]

print(f"Total issues: {len(issues)}")
print(f"Python issues: {len(python_issues)}")
print(f"With Dockerfiles: {len(dockerized)}")
```

### Running Evaluation

See [examples/USAGE_GUIDE.md](examples/USAGE_GUIDE.md) for detailed evaluation instructions.

```bash
# Quick evaluation on sample data
python -m cab_evaluation.cli single examples/sample_issue.json

# Batch evaluation
python -m cab_evaluation.cli dataset dataset/cab_verified.jsonl \
  --agent-models '{"maintainer": "sonnet37", "user": "haiku", "judge": "sonnet37"}'
```

## 📁 Project Structure

```
CodeAssistBench/
├── dataset/                    # 📊 Final datasets
│   ├── cab_recent.jsonl        # 308 recent issues
│   ├── cab_verified.jsonl      # 149 verified issues
│   └── recent/                 # Additional samples
├── src/cab_evaluation/         # 🔧 Evaluation framework
│   ├── agents/                 # Agent implementations
│   ├── core/                   # Core models and config
│   ├── prompts/                # Prompt templates
│   ├── utils/                  # Utilities
│   └── workflows/              # Evaluation workflows
├── script/                     # 🛠️ Data collection scripts
│   ├── get_github_issue.py     # Issue collection
│   ├── scon_filter.py          # Satisfaction condition generation
│   ├── docker_filter.py        # Dockerizability classification
│   └── generate_dockerfile_with_strands.py  # Dockerfile generation
├── tools/                      # Custom Strands tools
├── examples/                   # Sample data and guides
│   ├── USAGE_GUIDE.md          # Detailed usage guide
│   └── sample_*.jsonl          # Sample datasets
├── prompts/                    # Prompt templates
└── docs/                       # Documentation
    └── DATA_PIPELINE.md        # Data collection pipeline
```

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/CodeAssistBench.git
cd CodeAssistBench

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Install Strands tools
pip install -e tools/
```

### AWS Credentials (for evaluation)

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

## 📖 Documentation

- **[Usage Guide](examples/USAGE_GUIDE.md)** - Detailed evaluation instructions
- **[Data Pipeline](docs/DATA_PIPELINE.md)** - How the dataset was created
- **[Development](DEVELOPMENT.md)** - Contributing and development setup

## 🔄 Data Pipeline

The dataset was created through a multi-stage pipeline:

```
GitHub Issues → Filter & Commit → Satisfaction Conditions → Classification → Dockerfiles
```

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for:
- Complete pipeline documentation
- Scripts and commands used
- How to reproduce the dataset
- Example outputs at each stage

## 📚 Features

- **Automated Dockerfile Generation**: Uses Strands AI agents to generate Dockerfiles
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, Go, C, C++
- **Satisfaction Conditions**: LLM-generated criteria for issue resolution
- **Docker-based Evaluation**: Reproducible evaluation environment
- **Multiple Agent Frameworks**: Supports Strands, OpenHands, and Q-CLI

## 🛠️ Dockerfile Generation

Generate Dockerfiles for new issues:

```bash
STRANDS_NON_INTERACTIVE=true BYPASS_TOOL_CONSENT=true \
python script/generate_dockerfile_with_strands.py \
  --input-dir path/to/classified/issues \
  --max-attempts 3 \
  --parallel 4
```

See the [full documentation](#dockerfile-generation-options) for all options.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

The underlying GitHub issues are subject to their respective repository licenses.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Appendix: Dockerfile Generation Options

### Environment Variables

| Variable | Description |
|----------|-------------|
| `STRANDS_NON_INTERACTIVE=true` | Disables interactive prompts |
| `BYPASS_TOOL_CONSENT=true` | Bypasses tool confirmation |

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir`, `-i` | (required) | Directory with classified issues |
| `--output-dir`, `-o` | `logs/dockerfile_generation_strands` | Output directory |
| `--languages` | (all) | Specific languages to process |
| `--max-attempts` | `10` | Max retry attempts per issue |
| `--docker-timeout` | `600` | Docker build timeout (seconds) |
| `--agent-timeout` | `300` | Agent attempt timeout (seconds) |
| `--issue-timeout` | `1800` | Total timeout per issue (seconds) |
| `--parallel`, `-p` | `1` | Parallel processing count |
| `--model-id` | `claude-sonnet-4-5` | AWS Bedrock model ID |
