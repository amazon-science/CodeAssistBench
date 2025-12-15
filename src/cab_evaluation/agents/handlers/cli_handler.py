"""CLI-based custom agent handler."""

import json
import time
import logging
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime

from ...core.exceptions import AgentError

logger = logging.getLogger(__name__)


class CLIAgentHandler:
    """Handler for CLI-based custom agents.
    
    This handler communicates with external CLI tools by:
    1. Sending prompts and context via stdin as JSON
    2. Receiving responses via stdout as JSON or plain text
    3. Implementing retry logic with exponential backoff
    4. Validating and extracting metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CLI agent handler.
        
        Args:
            config: Configuration dictionary with:
                - command: CLI command to execute
                - args: List of command arguments
                - timeout: Command timeout in seconds
                - working_directory: Optional working directory
                - metrics: Metrics configuration
        """
        self.command = config.get("command")
        if not self.command:
            raise ValueError("CLI agent requires 'command' field in config")
        
        # Ensure args is always a list (handle None from dataclass conversion)
        self.args = config.get("args") or []
        if not isinstance(self.args, list):
            self.args = []
        
        self.timeout = config.get("timeout", 300)
        self.working_directory = config.get("working_directory")
        self.metrics_config = config.get("metrics") or {}
        self.max_retries = config.get("max_retries", 3)
        
        logger.info(f"CLI handler initialized: command={self.command}, timeout={self.timeout}s, retries={self.max_retries}")
    
    async def execute(
        self,
        prompt: str,
        repo_dir: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Execute CLI agent with retry logic.
        
        Args:
            prompt: User prompt
            repo_dir: Repository directory path
            system_prompt: System prompt for the agent
            
        Returns:
            Dictionary with:
                - response: Agent's response text
                - exploration_results: Optional exploration logs
                - metrics: Optional metrics data
                - success: Boolean indicating success
                
        Raises:
            AgentError: If execution fails after all retries
        """
        # Prepare input data
        input_data = {
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "repo_dir": repo_dir,
            "timestamp": datetime.now().isoformat()
        }
        
        input_json = json.dumps(input_data)
        
        # Working directory: prefer custom setting, then repo_dir
        working_dir = self.working_directory or repo_dir
        
        logger.info(f"Executing CLI agent: {self.command} {' '.join(self.args)}")
        logger.info(f"Working directory: {working_dir}")
        logger.info(f"Input prompt length: {len(prompt)} chars")
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Build full command
                full_command = [self.command] + self.args
                
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: Running CLI agent...")
                
                # Execute command with input via stdin
                result = subprocess.run(
                    full_command,
                    input=input_json,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=working_dir
                )
                
                elapsed_time = time.time() - start_time
                
                # Check return code
                if result.returncode == 0:
                    logger.info(f"✅ CLI agent succeeded in {elapsed_time:.2f}s")
                    
                    # Parse response
                    response_data = self._parse_response(result.stdout)
                    
                    # Validate metrics if configured
                    metrics = self._validate_metrics(response_data, self.metrics_config)
                    
                    return {
                        "response": response_data["response"],
                        "exploration_results": response_data.get("exploration", ""),
                        "metrics": metrics,
                        "success": True,
                        "execution_time": elapsed_time,
                        "attempt": attempt + 1
                    }
                else:
                    error_msg = result.stderr or "No error output"
                    logger.warning(
                        f"❌ CLI agent attempt {attempt + 1} failed "
                        f"(return code {result.returncode}): {error_msg[:200]}"
                    )
                    
                    # Retry with exponential backoff
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        logger.info(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"⏰ CLI agent timeout on attempt {attempt + 1} "
                    f"(timeout: {self.timeout}s)"
                )
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.warning(f"⚠️ CLI agent error on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise AgentError(
            f"CLI agent failed after {self.max_retries} attempts. "
            f"Last error: {result.stderr if result else 'Unknown error'}",
            agent_type="custom_cli"
        )
    
    def _parse_response(self, stdout: str) -> Dict[str, Any]:
        """Parse CLI agent stdout response.
        
        Supports both JSON and plain text responses.
        
        Args:
            stdout: Standard output from CLI agent
            
        Returns:
            Dictionary with response and optional fields
        """
        if not stdout or not stdout.strip():
            logger.warning("Empty stdout from CLI agent")
            return {"response": "", "exploration": ""}
        
        # Try to parse as JSON first
        try:
            data = json.loads(stdout)
            
            # Ensure required 'response' field exists
            if "response" not in data:
                logger.warning("CLI agent JSON missing 'response' field, using full output")
                return {"response": stdout, "exploration": ""}
            
            logger.info(f"✅ Parsed JSON response from CLI agent")
            return data
            
        except json.JSONDecodeError:
            # Fallback: treat as plain text response
            logger.info("CLI agent returned plain text (not JSON)")
            return {
                "response": stdout.strip(),
                "exploration": ""
            }
    
    def _validate_metrics(
        self,
        response_data: Dict[str, Any],
        metrics_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and extract metrics from response.
        
        Args:
            response_data: Parsed response from CLI agent
            metrics_config: Metrics configuration with expected fields
            
        Returns:
            Dictionary with validated metrics
        """
        if not metrics_config.get("enabled"):
            return {}
        
        metrics = response_data.get("metrics", {})
        
        # Validate expected metric fields
        expected_fields = metrics_config.get("expected_fields", [])
        missing_fields = []
        
        for field in expected_fields:
            if field not in metrics:
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(
                f"Custom agent metrics missing expected fields: {missing_fields}"
            )
        
        # Log metrics summary
        if metrics:
            logger.info(f"📊 Custom agent metrics received:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
        
        return metrics
