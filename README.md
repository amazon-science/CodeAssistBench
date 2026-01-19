# CodeAssistBench

A benchmark for evaluating AI coding assistants on real GitHub issues. This project includes a curated dataset of GitHub issues with Dockerfiles for reproducible evaluation, plus tools for dataset creation and AI agent evaluation.

## 📊 Dataset Overview

CodeAssistBench provides three ready-to-use datasets:

| Dataset | Issues | Languages | Description |
|---------|--------|-----------|-------------|
| `dataset/cab_recent_v2.jsonl` | 771 | 7 | **Latest** - June 2025 - Jan 2026 (with satisfaction conditions & classification) |
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

---

## 🛠️ Step-by-Step: Generate Your Own Dataset

This section walks through how we generated the dataset from scratch using **AWS Bedrock** and **Strands AI agents**.

### Prerequisites

```bash
# 1. Clone and setup
git clone https://github.com/your-org/CodeAssistBench.git
cd CodeAssistBench
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Install Strands SDK (required for Dockerfile generation)
pip install strands-agents strands-agents-tools
pip install -e tools/

# 4. Set up LLM credentials (choose ONE option)

# Option A: AWS Bedrock (Claude models)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2

# Option B: OpenAI (GPT-5 models)
export OPENAI_API_KEY=your_openai_api_key

# 5. Set up GitHub token (for API access)
export GITHUB_TOKEN=your_github_personal_access_token
```

### Step 1: Collect GitHub Issues

Collect closed issues from popular repositories. The script uses interactive prompts:

```bash
python script/get_github_issue.py
# Enter CSV path when prompted (see script/python_repos*.csv for examples)
# Choose label-based filtering (y/n)
```

Or use the bulk collection script:
```bash
python script/collect_1000_issues.py
# Edit the script to set: language, min_stars, date range
```

**Output:** `github_issues_<owner>_<repo>_<timestamp>.json`
```json
[
  {
    "number": 1234,
    "title": "Bug: Memory leak in parser",
    "url": "https://github.com/owner/repo/issues/1234",
    "body": "When parsing large files...",
    "comments": [...]
  }
]
```

### Step 2: Get Commit IDs

Find the commit hash at the time each issue was closed. Edit the script paths:

```bash
# Edit script/get_github_commit.py to set input/output paths, then run:
python script/get_github_commit.py
```

**Output:** Adds `commit_id` field to each issue.

### Step 3: Generate Satisfaction Conditions (Uses LLM)

Use LLM to generate explicit criteria for issue resolution. Edit the script paths:

```bash
# Edit script/scon_filter.py to set:
#   input_dir = "path/to/input"
#   output_dir = "path/to/output"
# Then run:
python script/scon_filter.py
```

**Configuration:** In `script/scon_filter.py`, modify the `BedrockConfig` class:
```python
@dataclass
class BedrockConfig:
    region: str = "us-west-2"
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Change for different model
```

**Output:** Adds `satisfaction_conditions` field:
```json
{
  "satisfaction_conditions": [
    "Memory usage remains stable when parsing files >100MB",
    "Parser handles all edge cases mentioned in the issue"
  ]
}
```

### Step 4: Classify Dockerizability (Uses LLM)

Classify issues by whether they need a Docker environment. Edit the script paths:

```bash
# Edit script/docker_filter.py to set input/output directories, then run:
python script/docker_filter.py
```

**Output:** Issues are categorized into:
- `classified/need_docker/` - Issues that need Docker
- `classified/no_need_docker/` - Documentation/config changes
- `classified/cannot_docker/` - Hardware-specific issues

### Step 5: Generate Dockerfiles (Uses Strands + LLM)

**⚠️ This step requires Strands AI agents** to automatically generate and test Dockerfiles:

```bash
# Option A: Using AWS Bedrock (Claude) - default
STRANDS_NON_INTERACTIVE=true BYPASS_TOOL_CONSENT=true \
python script/generate_dockerfile_with_strands.py \
  --input-dir my_data/classified/need_docker \
  --languages python \
  --max-attempts 3 \
  --parallel 2 \
  --agent-timeout 180 \
  --issue-timeout 600

# Option B: Using OpenAI (GPT-5)
STRANDS_NON_INTERACTIVE=true BYPASS_TOOL_CONSENT=true \
python script/generate_dockerfile_with_strands.py \
  --input-dir my_data/classified/need_docker \
  --languages python \
  --max-attempts 3 \
  --parallel 2 \
  --agent-timeout 180 \
  --issue-timeout 600 \
  --model-id gpt5 \
  --provider openai
```

**What happens:**
1. Strands agent reads the issue and repository structure
2. Agent generates a Dockerfile based on repo's build system
3. Docker builds the image to verify it works
4. If build fails, agent iterates with error feedback
5. Success: Dockerfile is saved to the issue JSON

**Output:** Adds `dockerfile` field:
```json
{
  "dockerfile": "FROM python:3.11-slim\n\nWORKDIR /workspace\n\nRUN apt-get update && apt-get install -y git\n\nRUN git clone https://github.com/owner/repo.git . && \\\n    git checkout abc123def456\n\nRUN pip install -r requirements.txt\n\nCMD [\"pytest\", \"tests/\"]\n"
}
```

### Step 6: Convert to Final Dataset

Combine all processed issues into a single JSONL file:

```bash
python script/convert_to_jsonl.py \
  --input-dir my_data/classified/need_docker \
  --output my_data/my_dataset.jsonl
```

---

## 📂 Example Outputs

See [`examples/`](examples/) for sample outputs at each pipeline stage:

| File | Description |
|------|-------------|
| `examples/sample_dataset.jsonl` | Complete issues with all fields |
| `examples/sample_docker_based_issues.jsonl` | Issues requiring Docker |
| `examples/sample_non_docker_based_issues.jsonl` | Documentation/config issues |
| `examples/sample_pipeline_output.json` | Single issue showing all fields |

---

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

---

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
│   ├── get_github_issue.py     # Step 1: Issue collection
│   ├── get_github_commit.py    # Step 2: Commit ID lookup
│   ├── scon_filter.py          # Step 3: Satisfaction conditions
│   ├── docker_filter.py        # Step 4: Classification
│   └── generate_dockerfile_with_strands.py  # Step 5: Dockerfiles
├── tools/                      # Custom Strands tools (required)
├── examples/                   # Sample data and guides
│   ├── USAGE_GUIDE.md          # Detailed usage guide
│   └── sample_*.jsonl          # Sample datasets
├── prompts/                    # Prompt templates
└── docs/                       # Documentation
    └── DATA_PIPELINE.md        # Detailed pipeline docs
```

---

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

# Install Strands SDK (REQUIRED for Dockerfile generation)
pip install strands-agents strands-agents-tools
pip install -e tools/
```

### AWS Credentials (Required for Bedrock)

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

---

## 📖 Documentation

- **[Usage Guide](examples/USAGE_GUIDE.md)** - Detailed evaluation instructions
- **[Data Pipeline](docs/DATA_PIPELINE.md)** - Complete pipeline documentation
- **[Development](DEVELOPMENT.md)** - Contributing and development setup

---

## 📚 Features

- **Automated Dockerfile Generation**: Uses Strands AI agents with AWS Bedrock
- **Multi-language Support**: Python, JavaScript, TypeScript, Java, Go, C, C++
- **Satisfaction Conditions**: LLM-generated criteria for issue resolution
- **Docker-based Evaluation**: Reproducible evaluation environment
- **Multiple Agent Frameworks**: Supports Strands, OpenHands, and Q-CLI

---

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
| `STRANDS_NON_INTERACTIVE=true` | **Required.** Disables interactive prompts |
| `BYPASS_TOOL_CONSENT=true` | **Required.** Bypasses tool confirmation |

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
