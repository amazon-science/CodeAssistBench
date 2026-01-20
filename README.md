# CodeAssistBench

A benchmark for evaluating AI coding assistants on real GitHub issues. This project includes a curated dataset of GitHub issues with Dockerfiles for reproducible evaluation, plus tools for dataset creation and AI agent evaluation.

## ⚡ Quick Run (5 minutes)

Get started immediately with our pre-built dataset:

```bash
# 1. Clone and install
git clone https://github.com/your-org/CodeAssistBench.git
cd CodeAssistBench
pip install -r requirements.txt && pip install -e .

# 2. Set AWS credentials (for Bedrock)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2

# 3. Run evaluation on 10 Python issues
python -m cab_evaluation.cli generation-dataset \
  dataset/cab_recent.jsonl \
  --output results/quick_test.jsonl \
  --agent-models '{"maintainer": "haiku", "user": "haiku"}' \
  --language python

# 4. Judge the results
python -m cab_evaluation.cli evaluation-dataset \
  results/quick_test.jsonl \
  --output results/quick_eval.jsonl \
  --agent-models '{"judge": "haiku"}'

# 5. View results
python -c "
import json
with open('results/quick_eval.jsonl') as f:
    for line in f:
        r = json.loads(line)
        print(f\"{r['issue_id']}: {r['verdict']}\")
"
```

**What this does:**
1. Generates maintainer responses for Python issues using Claude Haiku (fast & cheap)
2. Evaluates responses with a judge agent
3. Outputs verdicts: `CORRECT`, `PARTIALLY_CORRECT`, `INCORRECT`, or `ERROR`

For production evaluation, use `sonnet4` or `opus` models instead of `haiku`.

---

## 📊 Dataset Overview

CodeAssistBench provides three ready-to-use datasets:

| Dataset | Issues | Languages | Description |
|---------|--------|-----------|-------------|
| `dataset/cab_recent_v2.jsonl` | 771 | 7 | **Latest** - June 2025 - Jan 2026 (with satisfaction conditions & classification) |
| `dataset/cab_recent.jsonl` | 308 | 7 | Recent issues (June 2025 - Jan 2026) |
| `dataset/cab_verified.jsonl` | 149 | 7 | Verified subset with tested Dockerfiles |

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

Find the commit hash at the time each issue was closed:

```bash
python script/get_github_commit.py \
  --input-dir my_data/collected_issues \
  --output-dir my_data/with_commits

# Or using short options:
python script/get_github_commit.py -i my_data/collected_issues -o my_data/with_commits
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--input-dir`, `-i` | Yes | Directory containing JSON files with issues |
| `--output-dir`, `-o` | No | Output directory (default: `github_commits`) |

**Output:** Creates commit data files in the output directory.

### Step 3: Generate Satisfaction Conditions (Uses LLM)

Use LLM to generate explicit criteria for issue resolution:

```bash
python script/scon_filter.py \
  --input-dir my_data/collected_issues \
  --output-dir my_data/with_scon

# With custom model and region:
python script/scon_filter.py \
  -i my_data/collected_issues \
  -o my_data/with_scon \
  --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --region us-west-2
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir`, `-i` | Yes | - | Directory containing JSON files with issues |
| `--output-dir`, `-o` | Yes | - | Output directory for issues with satisfaction conditions |
| `--model`, `-m` | No | `claude-sonnet-4.5` | Bedrock model ID |
| `--region`, `-r` | No | `us-west-2` | AWS region for Bedrock |

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

Classify issues by whether they need a Docker environment:

```bash
python script/docker_filter.py \
  --input-dir my_data/with_scon \
  --output-dir my_data/classified

# With custom region:
python script/docker_filter.py \
  -i my_data/with_scon \
  -o my_data/classified \
  --region us-east-1
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir`, `-i` | Yes | - | Directory containing JSON files with issues |
| `--output-dir`, `-o` | Yes | - | Output directory for classified issues |
| `--region`, `-r` | No | `us-west-2` | AWS region for Bedrock |

**Output structure:**
```
my_data/classified/
├── need_docker/           # Issues that need Docker environment
├── no_need_docker/        # Documentation/config changes  
├── need_docker_but_cannot/ # Hardware-specific issues
├── llm_responses/         # Raw LLM responses for debugging
└── processed_issues.json  # Resume checkpoint
```

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

## 🧪 End-to-End Example

Here's a complete walkthrough processing 2 test issues through the entire pipeline:

### Setup

```bash
cd CodeAssistBench

# Set up credentials (AWS Bedrock + GitHub)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
export GITHUB_TOKEN=your_github_token
```

### Step 1: Create Test Data

Create a directory with sample issues:
```bash
mkdir -p test_pipeline/step1_raw
```

Create `test_pipeline/step1_raw/test_issues.json`:
```json
[
  {
    "number": 1234,
    "title": "How to handle async operations in Python?",
    "created_at": "2025-07-15T10:30:00Z",
    "url": "https://github.com/python/cpython/issues/1234",
    "body": "I'm trying to use async/await but get 'RuntimeWarning: coroutine was never awaited'.",
    "author": "user123",
    "comments": [
      {"user": "maintainer", "created_at": "2025-07-16T08:00:00Z", "body": "Use asyncio.run() to execute your coroutine."},
      {"user": "user123", "created_at": "2025-07-17T09:00:00Z", "body": "That worked perfectly!"}
    ]
  }
]
```

### Step 2: Generate Satisfaction Conditions

```bash
python3 script/scon_filter.py \
  --input-dir test_pipeline/step1_raw \
  --output-dir test_pipeline/step2_scon
```

**Expected output:**
```
Processing directory: test_pipeline/step1_raw
Found 1 JSON files
Processing conversation 1/1 (ID: 1234)
Added satisfaction conditions for conversation 1234
Saved 1 processed conversations to test_pipeline/step2_scon/test_issues.json
```

### Step 3: Classify Issues

```bash
python3 script/docker_filter.py \
  --input-dir test_pipeline/step2_scon \
  --output-dir test_pipeline/step3_classified
```

**Expected output:**
```
Input directory: test_pipeline/step2_scon
Output directory: test_pipeline/step3_classified
Found 1 JSON files to process.
Classified issue #1234 as: Does not need build environment

--- Classification Summary ---
Total issues processed: 1
Does not need build environment: 1 issues (100.0%)
```

### Final Directory Structure

```
test_pipeline/
├── step1_raw/
│   └── test_issues.json              # Original issues
├── step2_scon/
│   ├── test_issues.json              # + satisfaction_conditions
│   └── test_issues_prompts_responses.json
└── step3_classified/
    ├── no_need_docker/
    │   └── test_issues.json          # + _classification
    ├── need_docker/                   # (empty for this example)
    ├── llm_responses/                 # Raw LLM outputs
    └── classification_summary.json
```

### View Results

```bash
# Check satisfaction conditions were added
cat test_pipeline/step2_scon/test_issues.json | jq '.[0].satisfaction_conditions'

# Check classification
cat test_pipeline/step3_classified/no_need_docker/test_issues.json | jq '.[0]._classification'
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

The evaluation framework has **two phases**: Generation (maintainer answers issues) and Evaluation (judge scores responses).

#### Workflow Overview

```
┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐
│   Dataset       │ →  │   Generation   │ →  │   Evaluation    │
│   (JSONL)       │    │   Workflow     │    │   Workflow      │
└─────────────────┘    └────────────────┘    └─────────────────┘
                              ↓                      ↓
                       Maintainer ↔ User       Judge Agent
                       Multi-round chat        Scores answers
```

#### Step 1: Generation (Maintainer → User conversation)

```bash
python -m cab_evaluation.cli generation-dataset \
  dataset/cab_recent.jsonl \
  --output results/generation_results.jsonl \
  --agent-models '{"maintainer": "sonnet4", "user": "haiku"}' \
  --language python \
  --resume
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--output`, `-o` | Output file (default: auto-generated with timestamp) |
| `--agent-models` | JSON mapping models: `{"maintainer": "sonnet4", "user": "haiku"}` |
| `--language`, `-l` | Filter by language (python, javascript, etc.) |
| `--resume` | Skip already-processed issues |
| `--max-conversation-rounds` | Max rounds between maintainer/user (default: 2) |

#### Step 2: Evaluation (Judge scores responses)

```bash
python -m cab_evaluation.cli evaluation-dataset \
  results/generation_results.jsonl \
  --output results/evaluation_results.jsonl \
  --agent-models '{"judge": "sonnet4"}' \
  --resume
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--output`, `-o` | Output file for evaluation results |
| `--agent-models` | JSON with judge model: `{"judge": "sonnet4"}` |
| `--resume` | Skip already-evaluated issues |
| `--iterative` | Enable multi-iteration judge with repo exploration |

#### Verdict Types

The judge assigns one of these verdicts:

| Verdict | Description |
|---------|-------------|
| `CORRECT` | Response fully addresses the issue and satisfies all conditions |
| `PARTIALLY_CORRECT` | Response addresses some aspects but misses key elements |
| `INCORRECT` | Response doesn't address the issue or provides wrong information |
| `ERROR` | Processing failed (timeout, API error, etc.) |

#### Output Format

Each result in the JSONL file contains:

```json
{
  "issue_id": "1234",
  "question_title": "How to handle async operations?",
  "verdict": "CORRECT",
  "judgment": "The maintainer correctly identified the issue...",
  "key_issues": ["Clear explanation provided", "Code example included"],
  "alignment_score": {
    "satisfied": 3,
    "total": 3,
    "percentage": 100.0,
    "conditions": [
      {"number": 1, "satisfied": true, "description": "Explains async pattern"},
      {"number": 2, "satisfied": true, "description": "Provides working example"},
      {"number": 3, "satisfied": true, "description": "Addresses RuntimeWarning"}
    ]
  },
  "generation_metadata": {
    "user_satisfied": true,
    "total_conversation_rounds": 2
  }
}
```

#### Analyzing Results

```python
import json
from collections import Counter

# Load evaluation results
with open('results/evaluation_results.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# Count verdicts
verdicts = Counter(r['verdict'] for r in results)
print(f"Total: {len(results)}")
print(f"CORRECT: {verdicts['CORRECT']} ({verdicts['CORRECT']/len(results)*100:.1f}%)")
print(f"PARTIALLY_CORRECT: {verdicts['PARTIALLY_CORRECT']} ({verdicts['PARTIALLY_CORRECT']/len(results)*100:.1f}%)")
print(f"INCORRECT: {verdicts['INCORRECT']} ({verdicts['INCORRECT']/len(results)*100:.1f}%)")
print(f"ERROR: {verdicts.get('ERROR', 0)}")

# Average alignment score
valid_results = [r for r in results if r.get('alignment_score')]
avg_alignment = sum(r['alignment_score']['percentage'] for r in valid_results) / len(valid_results)
print(f"Average alignment: {avg_alignment:.1f}%")
```

#### Model Aliases

Available model shortcuts for `--agent-models`:

| Alias | Full Model ID |
|-------|---------------|
| `sonnet4` | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| `sonnet45` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `haiku` | `us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| `opus` | `us.anthropic.claude-opus-4-20250514-v1:0` |

See [examples/USAGE_GUIDE.md](examples/USAGE_GUIDE.md) for more detailed instructions.

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

## 📄 Citation

If you use CodeAssistBench in your research, please cite our paper:

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
