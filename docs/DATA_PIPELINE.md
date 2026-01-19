# CodeAssistBench Data Pipeline

This document describes the complete data pipeline used to create the CodeAssistBench dataset, from raw GitHub issue collection to the final dockerized evaluation dataset.

## Pipeline Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│  1. Collect     │───▶│  2. Filter &     │───▶│  3. Classify      │───▶│  4. Generate    │
│  GitHub Issues  │    │  Add Commits     │    │  Dockerizability  │    │  Dockerfiles    │
└─────────────────┘    └──────────────────┘    └───────────────────┘    └─────────────────┘
         │                      │                       │                       │
         ▼                      ▼                       ▼                       ▼
   raw JSON files        filtered JSON           classified JSON          final dataset
   (per language)        with commit IDs         with categories          with Dockerfiles
```

## Step 1: Collect GitHub Issues

**Script:** `script/get_github_issue.py`

Collects closed GitHub issues from popular repositories across multiple programming languages.

### Usage
```bash
python script/get_github_issue.py \
  --language python \
  --min-stars 1000 \
  --created-after 2025-06-01 \
  --created-before 2026-01-26 \
  --output-dir recent_v2_June2025_Jan2026
```

### Output Example
```json
{
  "number": 1234,
  "title": "Bug: Memory leak in parser",
  "created_at": "2025-07-15T10:30:00Z",
  "closed_at": "2025-07-20T14:22:00Z",
  "labels": ["bug", "parser"],
  "url": "https://github.com/owner/repo/issues/1234",
  "body": "When parsing large files, memory usage grows unbounded...",
  "author": "user123",
  "comments": [
    {
      "user": "maintainer",
      "created_at": "2025-07-16T08:00:00Z",
      "body": "Thanks for reporting! Can you share the file that triggers this?"
    }
  ]
}
```

### Languages Collected
| Language | Repository Count | Issues |
|----------|-----------------|--------|
| Python | ~500 repos | ~3.6M issues |
| JavaScript | ~500 repos | ~3.0M issues |
| TypeScript | ~500 repos | ~3.5M issues |
| Java | ~500 repos | ~2.6M issues |
| Go | ~500 repos | ~3.0M issues |
| C | ~500 repos | ~3.0M issues |
| C++ | ~500 repos | ~4.0M issues |

---

## Step 2: Filter & Add Commit Information

**Scripts:** 
- `script/get_github_commit.py` - Fetches commit data
- `script/merge_commit_ids.py` - Merges commit IDs into issues
- `script/filter_by_commit_date.py` - Filters by commit date

### Usage
```bash
# Get commits for each repository
python script/get_github_commit.py \
  --input recent_v2_June2025_Jan2026/python_issues.json \
  --output-dir script/github_commits

# Merge commit IDs into issues
python script/merge_commit_ids.py \
  --issues recent_v2_June2025_Jan2026/python_issues.json \
  --commits script/github_commits/
```

### Output
Adds `commit_id` field - the commit hash at the time the issue was closed:
```json
{
  "commit_id": "abc123def456789...",
  ...
}
```

---

## Step 3: Generate Satisfaction Conditions

**Script:** `script/scon_filter.py`

Uses LLM to analyze issues and generate explicit satisfaction conditions that define when an issue would be considered resolved.

### Usage
```bash
python script/scon_filter.py \
  --input recent_v2_June2025_Jan2026/python_issues.json \
  --output recent_v2_June2025_Jan2026_scon/python_issues.json \
  --model us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### Output Example
```json
{
  "satisfaction_conditions": [
    "Memory usage remains stable when parsing files larger than 100MB",
    "Parser correctly handles all edge cases mentioned in the issue",
    "No regression in parsing speed for normal-sized files"
  ],
  ...
}
```

---

## Step 4: Classify Dockerizability

**Script:** `script/docker_filter.py`

Uses LLM to classify issues based on whether they require a Docker build environment for evaluation.

### Usage
```bash
python script/docker_filter.py \
  --input recent_v2_June2025_Jan2026_scon/python_issues.json \
  --output-dir recent_v2_June2025_Jan2026_classified
```

### Classification Categories
| Category | Description |
|----------|-------------|
| `Can be dockerized without any issue` | Clear build/test steps, standard dependencies |
| `Needs Docker build environment` | Requires compilation, specific runtime |
| `Does not need build environment` | Documentation, config, or simple changes |
| `Cannot be dockerized` | Hardware-specific, requires external services |

### Output Example
```json
{
  "_classification": {
    "category": "Can be dockerized without any issue",
    "timestamp": "2025-04-14 01:01:54"
  },
  ...
}
```

---

## Step 5: Generate Dockerfiles (Strands Agent)

**Script:** `script/generate_dockerfile_with_strands.py`

Uses Strands AI agent to automatically generate Dockerfiles for issues that can be dockerized.

### Usage
```bash
STRANDS_NON_INTERACTIVE=true BYPASS_TOOL_CONSENT=true \
python script/generate_dockerfile_with_strands.py \
  --input-dir recent_v2_June2025_Jan2026_classified/need_docker \
  --max-attempts 3 \
  --parallel 7 \
  --agent-timeout 180 \
  --issue-timeout 1800
```

### Output Example
```json
{
  "dockerfile": "FROM python:3.11-slim\n\nWORKDIR /workspace\n\n# Install dependencies\nRUN apt-get update && apt-get install -y git\n\n# Clone repository at specific commit\nRUN git clone https://github.com/owner/repo.git . && \\\n    git checkout abc123def456789\n\n# Install Python dependencies\nRUN pip install -r requirements.txt\n\n# Set up test environment\nCMD [\"pytest\", \"tests/\"]\n",
  ...
}
```

---

## Step 6: Convert to Final Dataset

**Script:** `script/convert_to_jsonl.py`

Converts processed JSON files to JSONL format for the final dataset.

### Usage
```bash
python script/convert_to_jsonl.py \
  --input-dir recent_v2_June2025_Jan2026_final \
  --output dataset/cab_recent.jsonl
```

---

## Final Dataset Structure

### File: `dataset/cab_recent.jsonl`

Each line is a complete issue with all fields:

```json
{
  "number": 1234,
  "title": "Bug: Memory leak in parser",
  "created_at": "2025-07-15T10:30:00Z",
  "closed_at": "2025-07-20T14:22:00Z",
  "commit_id": "abc123def456789...",
  "labels": ["bug", "parser"],
  "url": "https://github.com/owner/repo/issues/1234",
  "body": "When parsing large files, memory usage grows unbounded...",
  "author": "user123",
  "comments": [
    {
      "user": "maintainer",
      "created_at": "2025-07-16T08:00:00Z",
      "body": "Thanks for reporting! Can you share the file that triggers this?"
    }
  ],
  "satisfaction_conditions": [
    "Memory usage remains stable when parsing files larger than 100MB",
    "Parser correctly handles all edge cases mentioned in the issue",
    "No regression in parsing speed for normal-sized files"
  ],
  "_classification": {
    "category": "Can be dockerized without any issue",
    "timestamp": "2025-04-14 01:01:54"
  },
  "dockerfile": "FROM python:3.11-slim\n...",
  "language": "python"
}
```

### Dataset Statistics

| Metric | cab_recent.jsonl | cab_verified.jsonl |
|--------|------------------|-------------------|
| Total Issues | 308 | 149 |
| Languages | 7 | 7 |
| With Dockerfiles | ~200 | 149 |
| Time Range | June 2025 - Jan 2026 | June 2025 - Jan 2026 |

### Language Distribution
| Language | Count |
|----------|-------|
| Python | ~50 |
| JavaScript | ~45 |
| TypeScript | ~50 |
| Java | ~40 |
| Go | ~45 |
| C | ~40 |
| C++ | ~38 |

---

## Reproducing the Pipeline

### Prerequisites
```bash
# Clone repository
git clone https://github.com/your-org/CodeAssistBench.git
cd CodeAssistBench

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up AWS credentials (for Bedrock LLM)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2

# Set up GitHub token (for API access)
export GITHUB_TOKEN=your_github_token
```

### Full Pipeline Execution
```bash
# 1. Collect issues (takes several hours)
python script/get_github_issue.py --language python --output-dir raw_data

# 2. Get commit information
python script/get_github_commit.py --input raw_data/python_issues.json

# 3. Generate satisfaction conditions
python script/scon_filter.py --input raw_data/python_issues.json --output scon_data/

# 4. Classify dockerizability
python script/docker_filter.py --input scon_data/python_issues.json --output-dir classified/

# 5. Generate Dockerfiles
STRANDS_NON_INTERACTIVE=true BYPASS_TOOL_CONSENT=true \
python script/generate_dockerfile_with_strands.py --input-dir classified/need_docker

# 6. Convert to final format
python script/convert_to_jsonl.py --input-dir final_data --output dataset/my_dataset.jsonl
```

---

## External Data Storage

Due to size constraints, intermediate pipeline data is hosted externally:

| Data | Location | Size |
|------|----------|------|
| Raw collected issues | [Request access] | ~26MB |
| LLM classification logs | [Request access] | ~11MB |
| Dockerfile generation logs | [Request access] | ~61MB |
| Full dataset with all metadata | HuggingFace (coming soon) | ~5MB |

---

## License

The CodeAssistBench dataset is released under [LICENSE]. The underlying GitHub issues are subject to their respective repository licenses.
