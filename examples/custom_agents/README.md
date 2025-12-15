# Custom Agents for CodeAssistBench

This directory contains example custom agents demonstrating the three supported integration types for CodeAssistBench.

## 🎯 Overview

CodeAssistBench supports custom maintainer agents through three interfaces:

1. **CLI** - External executable/script (Priority 1)
2. **Docker** - Containerized agent (Priority 2)
3. **Python Class** - Local Python module (Priority 3)

## 📋 Quick Start

### Prerequisites

```bash
# Ensure CodeAssistBench is installed
pip install -e .

# For CLI examples
chmod +x examples/custom_agents/example_cli_agent.py

# For Docker examples
docker --version  # Ensure Docker is running
```

---

## 🔧 1. CLI Agent

### What is it?
A standalone executable that communicates via stdin/stdout.

### Example: `example_cli_agent.py`

**Features:**
- Reads JSON input from stdin
- Outputs JSON response to stdout
- Supports retry logic (3 attempts)
- Reports custom metrics

**Usage:**
```bash
python -m cab_evaluation.cli generation-dataset examples/sample_dataset.jsonl \
  --agent-framework '{"maintainer": "custom"}' \
  --custom-agent-config '{
    "type": "cli",
    "command": "examples/custom_agents/example_cli_agent.py",
    "timeout": 300,
    "metrics": {
      "enabled": true,
      "expected_fields": ["input_tokens", "output_tokens", "cost"]
    }
  }'
```

### Expected Input (stdin):
```json
{
  "system_prompt": "You are a coding assistant...",
  "user_prompt": "How do I fix this bug?",
  "repo_dir": "/path/to/repository",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Expected Output (stdout):
```json
{
  "response": "Here's the solution...",
  "exploration": "Explored files: main.py, config.py",
  "metrics": {
    "execution_time": 2.5,
    "input_tokens": 1000,
    "output_tokens": 500,
    "cost": 0.015
  }
}
```

### Customization:
Edit `example_cli_agent.py` and replace the mock implementation with:
- Your CLI tool call (e.g., `claude-code`, `cursor`)
- Direct API calls (OpenAI, Anthropic, etc.)
- Local model inference

---

## 🐳 2. Docker Agent

### What is it?
A Docker container that runs your agent with isolated environment.

### Files:
- `Dockerfile.agent` - Container definition
- `docker_agent_wrapper.py` - Agent script inside container

**Features:**
- Repository mounted at `/workspace`
- Input via `CAB_INPUT` env var or stdin
- Isolated execution environment
- Full control over dependencies

### Build the Docker image:
```bash
docker build -f examples/custom_agents/Dockerfile.agent \
  -t cab-custom-agent:latest \
  examples/custom_agents/
```

### Usage:
```bash
python -m cab_evaluation.cli generation-dataset examples/sample_dataset.jsonl \
  --agent-framework '{"maintainer": "custom"}' \
  --custom-agent-config '{
    "type": "docker",
    "image": "cab-custom-agent:latest",
    "mount_repo": true,
    "timeout": 600,
    "metrics": {
      "enabled": true,
      "expected_fields": ["execution_time", "input_tokens", "cost"]
    }
  }'
```

### Customization:
1. Modify `Dockerfile.agent` to install your dependencies
2. Update `docker_agent_wrapper.py` with your agent logic
3. Rebuild the Docker image

### Use Cases:
- Local models (Ollama, llama.cpp)
- Specific Python/system dependencies
- GPU-accelerated inference
- Complete environment isolation

---

## 🐍 3. Python Class Agent

### What is it?
A Python class that implements the required interface methods.

### Example: `example_python_agent.py`

Contains two example classes:
- `MyCustomAgent` - Basic implementation
- `AdvancedCustomAgent` - With real API integration

**Features:**
- Direct Python integration
- No subprocess overhead
- Access to all Python libraries
- Optional metrics and exploration tracking

### Usage:
```bash
python -m cab_evaluation.cli generation-dataset examples/sample_dataset.jsonl \
  --agent-framework '{"maintainer": "custom"}' \
  --custom-agent-config '{
    "type": "python_class",
    "module_path": "examples/custom_agents/example_python_agent.py",
    "class_name": "MyCustomAgent",
    "init_args": {
      "api_key": "your-api-key",
      "model": "your-model-name"
    },
    "metrics": {
      "enabled": true,
      "expected_fields": ["input_tokens", "output_tokens", "total_cost"]
    }
  }'
```

### Required Methods:
```python
class YourCustomAgent:
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """REQUIRED: Generate response to user prompt."""
        pass
    
    def get_system_prompt(self, **kwargs) -> str:
        """REQUIRED: Return system prompt."""
        pass
```

### Optional Methods:
```python
    def get_metrics(self) -> Dict[str, Any]:
        """OPTIONAL: Return metrics for tracking."""
        pass
    
    def get_exploration_results(self) -> str:
        """OPTIONAL: Return exploration details."""
        pass
```

---

## 🎛️ Configuration Reference

### Common Configuration Fields

```json
{
  "type": "cli|docker|python_class",
  "timeout": 300,
  "max_retries": 3,
  "working_directory": "/custom/path",
  "metrics": {
    "enabled": true,
    "expected_fields": ["field1", "field2"]
  }
}
```

### CLI-Specific Fields
```json
{
  "type": "cli",
  "command": "/path/to/executable",
  "args": ["--flag1", "value1"]
}
```

### Docker-Specific Fields
```json
{
  "type": "docker",
  "image": "my-agent:latest",
  "mount_repo": true,
  "container_args": ["--flag1", "value1"]
}
```

### Python Class-Specific Fields
```json
{
  "type": "python_class",
  "module_path": "/path/to/agent.py",
  "class_name": "MyAgentClass",
  "init_args": {
    "api_key": "xxx",
    "model": "model-name"
  }
}
```

---

## 🔍 Testing Your Custom Agent

### Test CLI Agent:
```bash
# Test the CLI agent standalone
echo '{"user_prompt":"test","system_prompt":"test","repo_dir":"."}' | \
  python examples/custom_agents/example_cli_agent.py
```

### Test Docker Agent:
```bash
# Build and test Docker agent
docker build -f examples/custom_agents/Dockerfile.agent \
  -t cab-custom-agent:latest examples/custom_agents/

# Test the container
docker run --rm -e CAB_INPUT='{"user_prompt":"test","system_prompt":"test","repo_dir":"/workspace"}' \
  cab-custom-agent:latest
```

### Test Python Agent:
```python
# Test Python class agent
import asyncio
import sys
sys.path.append('examples/custom_agents')

from example_python_agent import MyCustomAgent

agent = MyCustomAgent(api_key="test", model="test-model")
response = asyncio.run(agent.generate_response(
    user_prompt="How do I fix this?",
    system_prompt="You are a helper",
    issue_id="test"
))
print(response)
```

---

## 🚨 Troubleshooting

### CLI Agent Issues

**Problem**: `Permission denied`
```bash
chmod +x examples/custom_agents/example_cli_agent.py
```

**Problem**: `Command not found`
- Use absolute path to your executable
- Ensure the command is in PATH

**Problem**: `JSON decode error`
- Verify your agent outputs valid JSON to stdout only
- Use stderr for debug logging

### Docker Agent Issues

**Problem**: `Image not found`
```bash
docker build -f examples/custom_agents/Dockerfile.agent \
  -t cab-custom-agent:latest examples/custom_agents/
```

**Problem**: `Container timeout`
- Increase timeout in config: `"timeout": 600`
- Check container logs: `docker logs <container_id>`

**Problem**: `Repository not accessible`
- Ensure `mount_repo: true` in config
- Repository is mounted at `/workspace` inside container

### Python Class Agent Issues

**Problem**: `Module not found`
- Use absolute path: `/full/path/to/your_agent.py`
- Verify file exists and is readable

**Problem**: `Class not found`
- Check class name matches exactly (case-sensitive)
- Ensure class is defined at module level

**Problem**: `Missing required methods`
- Implement `generate_response()` and `get_system_prompt()`
- Both methods are required

---

## 💡 Tips & Best Practices

### Performance
- Use appropriate timeouts (CLI: 300s, Docker: 600s, Python: 300s)
- Implement efficient retry logic
- Cache responses when possible

### Metrics
- Always include execution_time
- Track token usage if available
- Calculate costs for budget tracking

### Error Handling
- Return valid JSON even on errors
- Log errors to stderr (not stdout for CLI/Docker)
- Provide meaningful error messages

### Security
- Don't expose API keys in config files
- Use environment variables for secrets
- Validate input data before processing

---

## 📊 Metrics Format

### Recommended Metrics Fields
```json
{
  "metrics": {
    "execution_time": 2.5,
    "input_tokens": 1000,
    "output_tokens": 500,
    "total_tokens": 1500,
    "cost": 0.015,
    "model_used": "model-name",
    "cache_hit": false
  }
}
```

### Metric Validation
Configure expected fields to validate:
```json
{
  "metrics": {
    "enabled": true,
    "expected_fields": ["execution_time", "input_tokens", "output_tokens", "cost"]
  }
}
```

---

## 🚀 Real-World Examples

### Example 1: Wrap Claude Code CLI
```bash
# Create wrapper script
cat > my_claude_wrapper.sh << 'EOF'
#!/bin/bash
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.user_prompt')
claude-code --prompt "$PROMPT" | jq -R -s '{"response": ., "metrics": {}}'
EOF

chmod +x my_claude_wrapper.sh

# Use with CodeAssistBench
python -m cab_evaluation.cli generation-dataset examples/sample_dataset.jsonl \
  --agent-framework '{"maintainer": "custom"}' \
  --custom-agent-config '{"type": "cli", "command": "./my_claude_wrapper.sh"}'
```

### Example 2: Use Local Ollama Model
```python
# Create custom_ollama_agent.py
import ollama

class OllamaAgent:
    def __init__(self, model="llama2", **kwargs):
        self.model = model
    
    def get_system_prompt(self, **kwargs):
        return "You are a coding assistant."
    
    async def generate_response(self, user_prompt, system_prompt, issue_id="", **kwargs):
        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        return response['message']['content']
```

---

## 📚 Additional Resources

- Main documentation: `README.md`
- Usage guide: `examples/USAGE_GUIDE.md`
- Custom agents deep dive: `docs/CUSTOM_AGENTS.md` (coming soon)

---

## ✅ Checklist for Creating Your Custom Agent

- [ ] Choose interface type (CLI, Docker, or Python)
- [ ] Implement required interface (stdin/stdout or Python methods)
- [ ] Add metrics reporting (optional but recommended)
- [ ] Test standalone before integrating
- [ ] Configure timeout appropriately
- [ ] Add error handling and logging
- [ ] Test with CodeAssistBench on sample dataset
- [ ] Validate metrics are collected correctly

---

## 🤝 Contributing

Found a bug or have an improvement? Please contribute:
1. Test your custom agent implementation
2. Share successful integration patterns
3. Report issues or edge cases
4. Contribute additional example agents

---

## 📝 License

These examples are provided under the same license as CodeAssistBench. See LICENSE file for details.
