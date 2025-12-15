"""Python class-based custom agent handler."""

import json
import time
import logging
import sys
import importlib.util
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ...core.exceptions import AgentError

logger = logging.getLogger(__name__)


class PythonClassAgentHandler:
    """Handler for Python class-based custom agents.
    
    This handler loads and executes user-provided Python agent classes by:
    1. Dynamically importing the user's module
    2. Instantiating the agent class with provided arguments
    3. Calling standard interface methods (generate_response, etc.)
    4. Implementing retry logic with exponential backoff
    5. Validating and extracting metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Python class agent handler.
        
        Args:
            config: Configuration dictionary with:
                - module_path: Path to Python module file
                - class_name: Name of agent class to instantiate
                - init_args: Arguments to pass to agent __init__
                - metrics: Metrics configuration
        """
        self.module_path = config.get("module_path")
        if not self.module_path:
            raise ValueError("Python agent requires 'module_path' field in config")
        
        self.class_name = config.get("class_name")
        if not self.class_name:
            raise ValueError("Python agent requires 'class_name' field in config")
        
        # Ensure init_args is always a dict (handle None from dataclass conversion)
        self.init_args = config.get("init_args") or {}
        if not isinstance(self.init_args, dict):
            self.init_args = {}
        
        self.metrics_config = config.get("metrics") or {}
        self.max_retries = config.get("max_retries", 3)
        
        self._agent_instance = None
        self._module = None
        
        logger.info(f"Python handler initialized: module={self.module_path}, class={self.class_name}")
    
    def _load_agent_class(self):
        """Dynamically load and instantiate user's agent class.
        
        Returns:
            Instantiated agent object
            
        Raises:
            AgentError: If module loading or class instantiation fails
        """
        try:
            # Resolve absolute path
            module_path = Path(self.module_path).resolve()
            
            if not module_path.exists():
                raise AgentError(
                    f"Module file not found: {module_path}",
                    agent_type="custom_python"
                )
            
            # Load module from file
            module_name = f"custom_agent_{module_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            
            if spec is None:
                raise AgentError(
                    f"Could not load module spec from: {module_path}",
                    agent_type="custom_python"
                )
            
            self._module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = self._module
            spec.loader.exec_module(self._module)
            
            logger.info(f"✅ Module loaded: {module_name}")
            
            # Get agent class from module
            if not hasattr(self._module, self.class_name):
                raise AgentError(
                    f"Class '{self.class_name}' not found in module {module_path}",
                    agent_type="custom_python"
                )
            
            agent_class = getattr(self._module, self.class_name)
            
            # Validate required methods
            required_methods = ["generate_response", "get_system_prompt"]
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(agent_class, method):
                    missing_methods.append(method)
            
            if missing_methods:
                raise AgentError(
                    f"Custom agent class must implement methods: {missing_methods}. "
                    f"Required interface: generate_response(user_prompt, system_prompt, issue_id, **kwargs) -> str "
                    f"and get_system_prompt(**kwargs) -> str",
                    agent_type="custom_python"
                )
            
            logger.info(f"✅ Agent class validated: {self.class_name}")
            
            # Instantiate agent with user-provided init arguments
            logger.info(f"Instantiating agent with args: {list(self.init_args.keys())}")
            agent_instance = agent_class(**self.init_args)
            
            logger.info(f"✅ Agent instance created: {type(agent_instance).__name__}")
            
            return agent_instance
            
        except AgentError:
            raise
        except Exception as e:
            raise AgentError(
                f"Failed to load custom Python agent: {e}",
                agent_type="custom_python"
            )
    
    async def execute(
        self,
        prompt: str,
        repo_dir: str,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Execute Python class agent with retry logic.
        
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
        # Lazy load agent instance
        if self._agent_instance is None:
            self._agent_instance = self._load_agent_class()
        
        logger.info(f"Executing Python class agent: {self.class_name}")
        logger.info(f"Input prompt length: {len(prompt)} chars")
        
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: Calling agent.generate_response()...")
                
                # Call agent's generate_response method
                response = await self._agent_instance.generate_response(
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    issue_id="custom",
                    repo_dir=repo_dir
                )
                
                elapsed_time = time.time() - start_time
                
                if response:
                    logger.info(f"✅ Python agent succeeded in {elapsed_time:.2f}s")
                    
                    # Get metrics if agent provides them
                    metrics = {}
                    if hasattr(self._agent_instance, "get_metrics"):
                        try:
                            metrics = self._agent_instance.get_metrics()
                            metrics = self._validate_metrics(
                                {"metrics": metrics},
                                self.metrics_config
                            )
                        except Exception as e:
                            logger.warning(f"Could not get metrics from agent: {e}")
                    
                    # Get exploration results if available
                    exploration = ""
                    if hasattr(self._agent_instance, "get_exploration_results"):
                        try:
                            exploration = self._agent_instance.get_exploration_results()
                        except Exception as e:
                            logger.debug(f"No exploration results available: {e}")
                    
                    return {
                        "response": response,
                        "exploration_results": exploration,
                        "metrics": metrics,
                        "success": True,
                        "execution_time": elapsed_time,
                        "attempt": attempt + 1
                    }
                else:
                    logger.warning(f"⚠️ Python agent returned empty response on attempt {attempt + 1}")
                    last_error = "Empty response"
                    
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"⏳ Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                
            except Exception as e:
                logger.warning(f"⚠️ Python agent error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise AgentError(
            f"Python agent failed after {self.max_retries} attempts. "
            f"Last error: {last_error}",
            agent_type="custom_python"
        )
    
    def _validate_metrics(
        self,
        response_data: Dict[str, Any],
        metrics_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and extract metrics from response.
        
        Args:
            response_data: Response data with metrics
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
