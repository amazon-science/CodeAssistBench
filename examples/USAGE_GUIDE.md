# Strands Agent Usage Guide for CodeAssistBench

This guide shows how to use the Strands-enhanced CodeAssistBench system with the provided sample files.

## 🚨 Important: Virtual Environment Setup

Due to externally-managed Python environments, you MUST use a virtual environment:

```bash
# 1. Create virtual environment in CodeAssistBench directory
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install CodeAssistBench in development mode  
pip install -e .

# 5. Install Strands framework (if available locally)
# Navigate to Strands-Science directory and install:
pip install -e strands-1.12.0/
pip install -e tools/


# 7. Verify installation
python -c "from src.cab_evaluation.agents.agent_factory import AgentFactory; print('✅ Setup complete!')"
```

**Note**: Always activate the virtual environment before running CodeAssistBench:
```bash
source venv/bin/activate  # Run this each time you start a new terminal session
```

## Sample Files Included

1. **`sample_issue.json`** - Single issue for individual evaluation
2. **`sample_dataset.jsonl`** - Multiple issues for batch processing  
3. **`strands_agent_example.py`** - Code examples and demonstrations

## CLI Usage Examples

### 1. Single Issue Evaluation (Complete Workflow)

**Default model (sonnet37 for all agents):**
```bash
python -m cab_evaluation.cli single examples/sample_issue.json
```

**Custom model for all agents:**
```bash
python -m cab_evaluation.cli single examples/sample_issue.json \
  --agent-models '{"maintainer": "sonnet37", "user": "haiku", "judge": "sonnet37"}'
```

**Output:** `cab_result_<issue_id>.json`

### 2. Generation Workflow Only (Maintainer & User Agents)

**Default models:**
```bash
python -m cab_evaluation.cli generation examples/sample_issue.json
```

**Custom models for generation:**
```bash
python -m cab_evaluation.cli generation examples/sample_issue.json \
  --agent-models '{"maintainer": "sonnet37", "user": "haiku"}'
```

**Output:** `generation_result_<issue_id>.json`

### 3. Evaluation Workflow Only (Judge Agent)

**Default model:**
```bash
python -m cab_evaluation.cli evaluation generation_result_1234.json
```

**Custom model for evaluation:**
```bash
python -m cab_evaluation.cli evaluation generation_result_1234.json \
  --agent-models '{"judge": "sonnet37"}'
```

**Output:** `evaluation_result_<issue_id>.json`

### 4. Dataset Processing (All Agents)

**Process entire dataset:**
```bash
python -m cab_evaluation.cli dataset examples/sample_dataset.jsonl
```

**Filter by language and custom models:**
```bash
python -m cab_evaluation.cli dataset examples/sample_dataset.jsonl \
  --language python \
  --agent-models '{"maintainer": "sonnet37", "user": "sonnet37", "judge": "sonnet37"}'
```

**Batch processing with resume:**
```bash
python -m cab_evaluation.cli dataset examples/sample_dataset.jsonl \
  --batch-size 5 \
  --resume \
  --output-dir results_strands
```

## Model Selection Options

### Available Models
- `sonnet37` - Claude 3.7 Sonnet (default) - Best for complex reasoning
- `haiku` - Claude 3.5 Haiku - Fast and cost-effective
- `sonnet` - Claude 3.7 Sonnet (alias)
- `thinking` - Sonnet with thinking capabilities
- `deepseek` - DeepSeek R1 model
- `llama` - Meta Llama 3.3 70B

### Workflow-Specific Model Selection

**Generation Workflow (Maintainer + User):**
```json
{
  "maintainer": "sonnet37",  // Complex technical responses
  "user": "haiku"           // Faster user simulation
}
```

**Evaluation Workflow (Judge):**
```json
{
  "judge": "sonnet37"       // Detailed evaluation and reasoning
}
```

**Complete Workflow (All Agents):**
```json
{
  "maintainer": "sonnet37", // Technical problem solving
  "user": "haiku",         // User interaction simulation
  "judge": "sonnet37"      // Comprehensive evaluation
}
```

## Strands Framework Features

### Enhanced Tool Capabilities
All agents now have access to:
- **File Operations**: Read, write, and modify files
- **Command Execution**: Safe bash command execution
- **Repository Analysis**: Advanced code exploration
- **AWS Integration**: Direct AWS service interaction
- **Advanced Reasoning**: Enhanced thinking capabilities

### Performance Optimizations
- **Prompt Caching**: Automatic caching for cost reduction
- **Cost Tracking**: Detailed token usage and cost analysis
- **Metrics Collection**: Performance and efficiency monitoring

### Safety Controls
- **Read-Only Mode**: Safe operations only
- **Command Restrictions**: Built-in safety for bash execution
- **Graceful Fallback**: Works even without Strands framework

## Advanced Usage

### Configuration File
Create custom configuration:
```bash
python -m cab_evaluation.cli config examples/custom_config.json
```

### Read-Only Mode
For safe repository analysis:
```python
from cab_evaluation.agents.agent_factory import AgentFactory

factory = AgentFactory()
maintainer = factory.create_maintainer_agent(
    model_name="sonnet37",
    read_only=True  # Only safe operations
)
```

### Logging and Debugging
Enable detailed logging:
```bash
python -m cab_evaluation.cli single examples/sample_issue.json \
  --log-level DEBUG \
  --log-file strands_debug.log
```

## Sample File Structure

### Issue JSON Structure
```json
{
  "number": 1234,
  "title": "[Bug]: Issue description",
  "created_at": "ISO timestamp",
  "closed_at": "ISO timestamp", 
  "commit_id": "git commit hash",
  "labels": ["bug", "docker"],
  "url": "GitHub issue URL",
  "body": "Detailed issue description with markdown",
  "author": "username",
  "comments": [
    {
      "user": "maintainer_username",
      "created_at": "ISO timestamp",
      "body": "Response content with code examples"
    }
  ],
  "satisfaction_conditions": [
    "Condition 1: What the user needs to be satisfied",
    "Condition 2: Technical requirements",
    "Condition 3: Additional expectations"
  ],
  "_classification": {
    "category": "Needs Docker build environment | Does not need build environment",
    "timestamp": "completion timestamp"
  }
}
```

### Dataset JSONL Structure
Each line contains one complete issue JSON object as shown above.

## Expected Results

### Generation Result Structure
```json
{
  "issue_id": "1234",
  "user_satisfied": true/false,
  "satisfaction_status": "FULLY_SATISFIED|PARTIALLY_SATISFIED|NOT_SATISFIED",
  "total_conversation_rounds": 3,
  "conversation_history": [...],
  "agent_model_mapping": {...}
}
```

### Evaluation Result Structure  
```json
{
  "issue_id": "1234",
  "verdict": "CORRECT|PARTIALLY_CORRECT|INCORRECT",
  "alignment_score": {
    "satisfied": 3,
    "total": 3,
    "percentage": 100.0
  },
  "docker_results": {...},
  "agent_model_mapping": {...}
}
```

## Performance Tips

1. **Use appropriate models**: sonnet37 for complex reasoning, haiku for speed
2. **Enable caching**: Automatic with Strands framework
3. **Monitor costs**: Check logs for token usage and cost tracking
4. **Batch processing**: Use reasonable batch sizes for large datasets
5. **Resume capability**: Use `--resume` for interrupted dataset processing

## Troubleshooting

### Common Issues
- **Import errors**: Ensure Strands framework is available in Python path
- **Model access**: Verify AWS credentials for Bedrock model access  
- **Permission errors**: Check file permissions for read/write operations
- **Tool failures**: Review logs for specific tool execution errors

### Fallback Behavior
If Strands framework is unavailable, agents automatically fall back to standard LLM service with reduced capabilities but full compatibility.

## Next Steps

1. Try the sample files with different model combinations
2. Monitor the enhanced logging and metrics
3. Experiment with read-only mode for safe operations
4. Use the tool capabilities for repository analysis and code exploration
