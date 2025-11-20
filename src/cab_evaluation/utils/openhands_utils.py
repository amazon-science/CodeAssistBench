"""Utility functions for OpenHands integration."""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def validate_openhands_config(model_name: str) -> str:
    """Validate OpenHands configuration and return API key.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        API key from environment
        
    Raises:
        ValueError: If validation fails
    """
    # Validate model name
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"Invalid model name: {model_name}. Must be a non-empty string.")
    
    # Validate API key
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No LLM API key found. Set LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY"
        )
    
    return api_key


def validate_openhands_installation() -> bool:
    """Check if OpenHands SDK is installed."""
    try:
        # Check for SDK components
        from openhands.sdk import Agent, Conversation, LLM, Tool
        from openhands.tools.terminal import TerminalTool
        from openhands.tools.file_editor import FileEditorTool
        
        logger.info("✅ OpenHands SDK installation validated")
        return True
        
    except ImportError as e:
        logger.warning(f"OpenHands SDK validation failed: {e}")
        return False


def get_openhands_version() -> Optional[str]:
    """Get installed OpenHands SDK version."""
    try:
        from openhands.sdk import version
        ver = version()
        logger.info(f"OpenHands SDK version: {ver}")
        return ver
    except ImportError:
        logger.debug("OpenHands SDK not importable")
        return None
    except Exception:
        # Fallback to package version
        try:
            import openhands
            ver = getattr(openhands, '__version__', 'unknown')
            return ver
        except:
            return None


def create_openhands_config(
    output_path: str,
    model: str = "anthropic/claude-3-5-sonnet-20241022",
    workspace_base: Optional[str] = None,
    timeout: int = 1800
) -> str:
    """Create OpenHands configuration file."""
    if workspace_base is None:
        workspace_base = "/tmp/openhands_workspace"
    
    config_content = f"""[core]
workspace_base = "{workspace_base}"
cache_dir = "{workspace_base}/cache"
run_as_openhands = false
sandbox_type = "local"

[llm]
model = "{model}"
api_key_env = "ANTHROPIC_API_KEY"
temperature = 0.0
top_p = 1.0
max_tokens = 8192

[agent]
name = "CodeActAgent"
memory_enabled = false

[sandbox]
runtime_container_image = "nikolaik/python-nodejs:python3.11-nodejs22"
user_id = 1000
timeout = {timeout}
enable_auto_lint = false

[security]
confirmation_mode = false
security_analyzer = "local"
"""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(config_content)
    
    logger.info(f"Created OpenHands config: {output_path}")
    return str(output_file)


def parse_openhands_response_file(response_file: str) -> Dict[str, Any]:
    """Parse OpenHands response file."""
    try:
        with open(response_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            return {
                "response": content,
                "type": "text"
            }
            
    except Exception as e:
        logger.error(f"Error parsing OpenHands response file: {e}")
        return {
            "response": "",
            "error": str(e)
        }


def cleanup_openhands_workspace(workspace_dir: str, preserve_outputs: bool = True):
    """Clean up OpenHands workspace directory."""
    workspace_path = Path(workspace_dir)
    
    if not workspace_path.exists():
        logger.debug(f"Workspace directory does not exist: {workspace_dir}")
        return
    
    try:
        if preserve_outputs:
            output_files = [
                "code_assessment_report.json",
                "trajectory_summary.md",
                "agent_history.json"
            ]
            
            parent = workspace_path.parent
            for output_file in output_files:
                src = workspace_path / output_file
                if src.exists():
                    dst = parent / f"{workspace_path.name}_{output_file}"
                    shutil.move(str(src), str(dst))
                    logger.info(f"Preserved output: {dst}")
        
        shutil.rmtree(workspace_path)
        logger.info(f"🧹 Cleaned up workspace: {workspace_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to cleanup workspace {workspace_dir}: {e}")


def check_openhands_dependencies() -> Dict[str, bool]:
    """Check OpenHands SDK and dependencies availability."""
    status = {}
    
    try:
        from openhands.sdk import Agent, Conversation, LLM
        status['openhands_sdk'] = True
    except ImportError:
        status['openhands_sdk'] = False
    
    try:
        from openhands.tools.terminal import TerminalTool
        from openhands.tools.file_editor import FileEditorTool
        status['openhands_tools'] = True
    except ImportError:
        status['openhands_tools'] = False
    
    status['llm_api_key'] = 'LLM_API_KEY' in os.environ
    status['anthropic_api_key'] = 'ANTHROPIC_API_KEY' in os.environ
    status['openai_api_key'] = 'OPENAI_API_KEY' in os.environ
    status['has_api_key'] = any([
        status['llm_api_key'],
        status['anthropic_api_key'],
        status['openai_api_key']
    ])
    
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            timeout=5
        )
        status['docker'] = result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        status['docker'] = False
    
    return status


def get_openhands_status_report() -> str:
    """Get formatted OpenHands SDK installation status."""
    status = check_openhands_dependencies()
    
    report = "OpenHands SDK Installation Status:\n"
    report += "=" * 40 + "\n"
    
    if status['openhands_sdk']:
        version = get_openhands_version()
        report += f"✅ OpenHands SDK: Installed (v{version})\n"
    else:
        report += "❌ OpenHands SDK: Not installed\n"
        report += "   Install with: pip install openhands-sdk\n"
    
    if status['openhands_tools']:
        report += "✅ OpenHands Tools: Installed\n"
    else:
        report += "❌ OpenHands Tools: Not installed\n"
        report += "   Install with: pip install openhands-tools\n"
    
    if status['has_api_key']:
        if status['llm_api_key']:
            report += "✅ LLM API Key: Set (LLM_API_KEY)\n"
        elif status['anthropic_api_key']:
            report += "✅ API Key: Set (ANTHROPIC_API_KEY)\n"
        elif status['openai_api_key']:
            report += "✅ API Key: Set (OPENAI_API_KEY)\n"
    else:
        report += "⚠️  No API Key: Not set\n"
        report += "   Set one of: LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY\n"
    
    if status['docker']:
        report += "✅ Docker: Available (optional)\n"
    else:
        report += "⚠️  Docker: Not available (optional for sandbox)\n"
    
    report += "=" * 40
    
    return report


def map_cab_model_to_openhands(cab_model: str) -> str:
    """Map CAB model names to OpenHands model identifiers."""
    import re
    
    model_mapping = {
        "sonnet37": "claude-3-5-sonnet",
        "sonnet": "claude-3-5-sonnet",
        "haiku": "claude-3-5-haiku",
        "thinking": "claude-3-5-sonnet",
        "claude-3.5-sonnet": "claude-3-5-sonnet",
        "claude-3-5-sonnet": "claude-3-5-sonnet",
        "claude-3.5-haiku": "claude-3-5-haiku",
        "claude-3-5-haiku": "claude-3-5-haiku",
        "claude-3-opus": "claude-3-opus",
        "claude-3-sonnet": "claude-3-sonnet",
        "claude-3-haiku": "claude-3-haiku",
        "gpt-4": "gpt-4-turbo-preview",
        "gpt-4-turbo": "gpt-4-turbo-preview",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
    }
    
    if "/" in cab_model:
        _, model_part = cab_model.split("/", 1)
        model_part = model_part.replace("claude-3.5-", "claude-3-5-")
        model_part = re.sub(r'-\d{8}$', '', model_part)
        return model_part
    
    mapped = model_mapping.get(cab_model.lower())
    
    if mapped:
        if cab_model.lower() in ["sonnet37", "sonnet", "thinking"]:
            logger.info(f"Mapped '{cab_model}' → '{mapped}' (using Claude 3.5 for OpenHands compatibility)")
        return mapped
    else:
        logger.warning(f"Unknown model '{cab_model}', using as-is")
        result = cab_model.replace("claude-3.5-", "claude-3-5-")
        result = re.sub(r'-\d{8}$', '', result)
        return result
