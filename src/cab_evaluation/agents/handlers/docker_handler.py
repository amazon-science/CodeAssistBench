"""Docker-based custom agent handler."""

import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ...core.exceptions import AgentError

logger = logging.getLogger(__name__)


class DockerAgentHandler:
    """Handler for Docker-based custom agents.
    
    This handler communicates with Docker container agents by:
    1. Running container with mounted repository
    2. Sending prompts via environment variable or stdin
    3. Receiving responses via stdout
    4. Implementing retry logic with exponential backoff
    5. Validating and extracting metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Docker agent handler.
        
        Args:
            config: Configuration dictionary with:
                - image: Docker image name
                - mount_repo: Whether to mount repository (default True)
                - container_args: List of container arguments
                - timeout: Container execution timeout in seconds
                - metrics: Metrics configuration
        """
        self.image = config.get("image")
        if not self.image:
            raise ValueError("Docker agent requires 'image' field in config")
        
        self.mount_repo = config.get("mount_repo", True)
        self.container_args = config.get("container_args", [])
        self.timeout = config.get("timeout", 600)
        self.metrics_config = config.get("metrics", {})
        self.max_retries = config.get("max_retries", 3)
        
        # Validate Docker availability
        self._validate_docker()
        
        logger.info(f"Docker handler initialized: image={self.image}, timeout={self.timeout}s, retries={self.max_retries}")
    
    def _validate_docker(self):
        """Validate Docker availability."""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            logger.info("✅ Docker client validated")
        except ImportError:
            raise AgentError(
                "Docker package not installed. Install with: pip install docker",
                agent_type="custom_docker"
            )
        except Exception as e:
            raise AgentError(
                f"Docker not available: {e}. Ensure Docker daemon is running.",
                agent_type="custom_docker"
            )
    
    async def execute(
        self,
        prompt: str,
        repo_dir: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Execute Docker agent with retry logic.
        
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
        import docker
        
        # Prepare input data
        input_data = {
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "repo_dir": "/workspace",  # Path inside container
            "timestamp": datetime.now().isoformat()
        }
        
        input_json = json.dumps(input_data)
        
        logger.info(f"Executing Docker agent: {self.image}")
        logger.info(f"Repository mount: {self.mount_repo}")
        logger.info(f"Input prompt length: {len(prompt)} chars")
        
        client = docker.from_env()
        
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            container = None
            try:
                start_time = time.time()
                
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: Starting Docker container...")
                
                # Prepare volumes
                volumes = {}
                if self.mount_repo:
                    volumes[repo_dir] = {"bind": "/workspace", "mode": "rw"}
                
                # Run container
                container = client.containers.run(
                    self.image,
                    command=self.container_args if self.container_args else None,
                    volumes=volumes,
                    stdin_open=True,
                    detach=True,
                    remove=False,  # Don't auto-remove so we can get logs
                    environment={"CAB_INPUT": input_json}
                )
                
                logger.info(f"Container started: {container.short_id}")
                
                # Send input via stdin (alternative to ENV)
                try:
                    container_socket = container.attach_socket(
                        params={"stdin": 1, "stream": 1}
                    )
                    container_socket._sock.sendall(input_json.encode() + b'\n')
                    container_socket.close()
                except Exception as e:
                    logger.debug(f"Could not send via stdin (using ENV fallback): {e}")
                
                # Wait for completion with timeout
                result = container.wait(timeout=self.timeout)
                
                elapsed_time = time.time() - start_time
                
                # Get output
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='ignore')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='ignore')
                
                # Check exit code
                exit_code = result.get("StatusCode", -1)
                
                if exit_code == 0:
                    logger.info(f"✅ Docker agent succeeded in {elapsed_time:.2f}s")
                    
                    # Parse response
                    response_data = self._parse_response(stdout)
                    
                    # Validate metrics if configured
                    metrics = self._validate_metrics(response_data, self.metrics_config)
                    
                    # Cleanup container
                    container.remove()
                    
                    return {
                        "response": response_data["response"],
                        "exploration_results": response_data.get("exploration", ""),
                        "metrics": metrics,
                        "success": True,
                        "execution_time": elapsed_time,
                        "attempt": attempt + 1
                    }
                else:
                    error_msg = stderr or "No error output"
                    logger.warning(
                        f"❌ Docker agent attempt {attempt + 1} failed "
                        f"(exit code {exit_code}): {error_msg[:200]}"
                    )
                    last_error = error_msg
                    
                    # Cleanup failed container
                    if container:
                        container.remove()
                    
                    # Retry with exponential backoff
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        logger.info(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    
            except Exception as e:
                logger.warning(f"⚠️ Docker agent error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                
                # Cleanup container on error
                if container:
                    try:
                        container.remove(force=True)
                    except:
                        pass
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise AgentError(
            f"Docker agent failed after {self.max_retries} attempts. "
            f"Last error: {last_error}",
            agent_type="custom_docker"
        )
    
    def _parse_response(self, stdout: str) -> Dict[str, Any]:
        """Parse Docker agent stdout response.
        
        Supports both JSON and plain text responses.
        
        Args:
            stdout: Standard output from Docker container
            
        Returns:
            Dictionary with response and optional fields
        """
        if not stdout or not stdout.strip():
            logger.warning("Empty stdout from Docker agent")
            return {"response": "", "exploration": ""}
        
        # Try to parse as JSON first
        try:
            data = json.loads(stdout)
            
            # Ensure required 'response' field exists
            if "response" not in data:
                logger.warning("Docker agent JSON missing 'response' field, using full output")
                return {"response": stdout, "exploration": ""}
            
            logger.info(f"✅ Parsed JSON response from Docker agent")
            return data
            
        except json.JSONDecodeError:
            # Fallback: treat as plain text response
            logger.info("Docker agent returned plain text (not JSON)")
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
            response_data: Parsed response from Docker agent
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
