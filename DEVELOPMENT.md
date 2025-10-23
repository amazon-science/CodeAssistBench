# Development Guide

This document provides comprehensive information about the scripts available in the `script/` folder for contributors working with the CodeAssistBench dataset.

## Overview

The `script/` folder contains production-ready, configurable tools designed for public use. These scripts are built with flexibility in mind to work across different environments, programming languages, and folder structures.

## Available Scripts

### 1. generate_dockerfile.py

**Purpose**: Generate Dockerfiles for GitHub issues that are classified as dockerizable but missing dockerfile information.

**Description**: 
This script uses AWS Bedrock (Claude 3.7 Sonnet) to generate comprehensive Dockerfiles for GitHub issues. It includes advanced features like iterative improvement, Docker build testing, parallel processing, and failure analysis. The generated Dockerfiles are designed to create environments where the GitHub issues can be reproduced and validated.

**Key Features**:
- Configurable input/output paths for different environments
- Support for multiple programming languages with filtering
- LLM-powered Dockerfile generation with adaptive context management
- Docker build testing with automatic cleanup
- Iterative improvement based on build failures
- Parallel candidate generation and testing
- Reference Dockerfile usage for consistency within repositories
- Comprehensive logging and failure analysis

**Usage**:

```bash
# Basic usage with minimal configuration
python script/generate_dockerfile.py --input-dir dataset/updated

# Production usage with full customization
python script/generate_dockerfile.py \
    --input-dir /path/to/dataset \
    --output-dir /path/to/results \
    --log-dir /path/to/logs \
    --failure-logs-dir /path/to/failure_logs \
    --languages python javascript typescript \
    --dockerizable-category "Can be dockerized without any issue" \
    --candidates 5 \
    --max-attempts 3 \
    --disable-docker-testing \
    --aws-region us-west-2 \
    --model-id "us.anthropic.claude-3-5-sonnet-20241022-v2:0" \
    --max-workers 10 \
    --verbose

# Research environment usage
python script/generate_dockerfile.py \
    --input-dir research_dataset \
    --dockerizable-category "Research-ready issues" \
    --candidates 3 \
    --disable-docker-testing
```

**Command Line Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir, -i` | string | **Required** | Input directory containing GitHub issues dataset |
| `--output-dir, -o` | string | `logs/dockerfile_generation` | Output directory for generated results |
| `--log-dir` | string | `logs` | Directory for log files |
| `--failure-logs-dir` | string | `logs/dockerfile_failures` | Directory for build failure logs |
| `--languages` | list | all | Specific programming languages to process |
| `--dockerizable-category` | string | `Can be dockerized without any issue` | Classification category for dockerizable issues |
| `--candidates` | integer | 3 | Number of Dockerfile candidates to generate per issue |
| `--max-attempts` | integer | 4 | Maximum improvement attempts for failed Dockerfiles |
| `--disable-docker-testing` | flag | false | Disable Docker build testing |
| `--docker-timeout` | integer | 600 | Docker build timeout in seconds |
| `--aws-region` | string | `us-east-2` | AWS region for Bedrock API calls |
| `--model-id` | string | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | AWS Bedrock model ID |
| `--max-workers` | integer | 5 | Maximum number of parallel workers |
| `--verbose, -v` | flag | false | Enable verbose logging output |

**Prerequisites**:
- **AWS Credentials**: Properly configured AWS credentials for Bedrock access
- **Docker** (optional): Docker installed and daemon running for build testing
- **Python Dependencies**: `boto3`, `PyYAML`, `concurrent.futures`

**Output Files**:
- **Dataset Updates**: Original dataset files updated with dockerfile columns
- **Generation Logs**: `{log_dir}/dockerfile_generation_{timestamp}.log`
- **Failure Analysis**: `{failure_logs_dir}/{repo}_{issue}_{timestamp}_failure.json`

**Use Cases**:
- **Dataset Enhancement**: Add missing dockerfiles to research datasets
- **CI/CD Integration**: Automated dockerfile generation in pipelines  
- **Research Projects**: Generate dockerfiles for specific language subsets
- **Quality Assurance**: Validate dockerfiles with build testing

---

### 2. update_dataset_with_commits.py

**Purpose**: Update GitHub issues dataset with commit information by fetching commit IDs from GitHub API.

**Description**:
This script enhances GitHub issues datasets by adding commit information. For each issue, it finds the most recent commit in the repository that occurred before the issue's creation date. This provides crucial context about the repository state when the issue was created.

**Key Features**:
- Threaded GitHub API calls for parallel processing
- Intelligent commit selection based on issue creation timestamps
- Rate limit handling with both authenticated and unauthenticated modes
- Proper field ordering in output JSON files
- Comprehensive error handling and logging
- Resume capability for interrupted processing

**Usage**:

```bash
# Basic usage with default settings
python script/update_dataset_with_commits.py

# With GitHub token for higher rate limits
python script/update_dataset_with_commits.py \
    --github-token YOUR_GITHUB_TOKEN

# Full customization
python script/update_dataset_with_commits.py \
    --source-dir dataset/raw_issues \
    --target-dir dataset/with_commits \
    --threads 15 \
    --github-token YOUR_GITHUB_TOKEN

# High-throughput processing
python script/update_dataset_with_commits.py \
    --threads 20 \
    --github-token YOUR_TOKEN
```

**Command Line Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--github-token, -t` | string | None | GitHub personal access token for API authentication |
| `--source-dir` | string | `dataset/recent` | Source dataset directory |
| `--target-dir` | string | `dataset/updated` | Target directory for updated dataset |
| `--threads, -j` | integer | 10 | Number of threads for parallel processing |

**Prerequisites**:
- **GitHub Token** (recommended): Personal access token for higher API rate limits
- **Internet Connection**: Access to GitHub API
- **Python Dependencies**: `requests`, `concurrent.futures`

**Rate Limits**:
- **Without Token**: 60 requests/hour per IP address
- **With Token**: 5,000 requests/hour per token
- **Recommended**: Use GitHub token for processing large datasets

**Output**:
- **Updated Dataset**: New directory with commit_id fields added
- **Processing Logs**: Detailed logs of API calls and processing status
- **Error Reports**: Information about failed repository lookups

**Use Cases**:
- **Dataset Preparation**: Add temporal context to GitHub issues
- **Research Analysis**: Study issue-commit relationships
- **Data Pipeline**: Automated dataset enhancement in CI/CD
- **Quality Control**: Ensure all issues have associated commit information

## Environment Setup

### AWS Configuration
```bash
# Method 1: AWS CLI configuration
aws configure

# Method 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-2

# Method 3: IAM roles (for EC2/ECS environments)
# Attach appropriate IAM role with Bedrock permissions
```

### GitHub API Setup
```bash
# Create personal access token at: https://github.com/settings/tokens
# Set environment variable
export GITHUB_TOKEN=ghp_your_token_here

# Or pass directly to scripts
--github-token ghp_your_token_here
```

### Docker Setup (for generate_dockerfile.py)
```bash
# Install Docker Desktop
# Verify installation
docker --version
docker info

# Alternative: Disable testing
--disable-docker-testing
```

## Best Practices for Contributors

### 1. Configuration Management
- **Always use command-line arguments** instead of modifying script constants
- **Provide default values** for optional parameters
- **Validate input paths** before processing
- **Create output directories** automatically

### 2. Error Handling
- **Implement comprehensive logging** for debugging
- **Handle API rate limits** gracefully
- **Provide clear error messages** with actionable guidance
- **Support resume/retry** for long-running operations

### 3. Performance Optimization
- **Use parallel processing** where appropriate
- **Implement early termination** for successful cases
- **Cache successful results** for reuse
- **Optimize API usage** to minimize costs

### 4. Documentation
- **Include comprehensive docstrings** for all functions
- **Provide usage examples** in help text
- **Document prerequisites** and setup requirements
- **Explain output formats** and file structures

## Testing and Validation

### Script Validation
```bash
# Test help system
python script/generate_dockerfile.py --help
python script/update_dataset_with_commits.py --help

# Validate with small datasets first
python script/generate_dockerfile.py \
    --input-dir test_dataset \
    --candidates 1 \
    --disable-docker-testing
```

### Environment Testing
```bash
# Test AWS connectivity
aws sts get-caller-identity

# Test GitHub API access
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

# Test Docker functionality
docker run hello-world
```

## Support and Troubleshooting

### Common Error Messages

**"AWS credentials not configured"**
- Configure AWS CLI: `aws configure`
- Set environment variables for access keys
- Verify IAM permissions for Bedrock service

**"GitHub API rate limit exceeded"**
- Use GitHub personal access token: `--github-token YOUR_TOKEN`
- Reduce thread count: `--threads 5`
- Wait for rate limit reset (1 hour for unauthenticated)

**"Docker daemon not running"**
- Start Docker Desktop application
- Use `--disable-docker-testing` to skip Docker functionality
- Verify Docker installation: `docker --version`

**"Directory not found"**
- Verify input directory path exists
- Use absolute paths if relative paths cause issues
- Check file permissions for read/write access

### Getting Help
1. **Use `--help`** flag for comprehensive usage information
2. **Check log files** for detailed error traces and LLM interactions
3. **Validate prerequisites** before running scripts
4. **Start with small test datasets** to verify functionality

For additional support, please refer to the project's issue tracker or contact the development team.
