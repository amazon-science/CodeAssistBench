"""CodeAssistBench Package.

A comprehensive package for CodeAssistBench (CAB) evaluation providing
generation, evaluation, and complete CAB workflows for agent testing.
"""

from .core.config import CABConfig, DEFAULT_CONFIG
from .core.models import (
    IssueData,
    GenerationResult,
    EvaluationResult,
    CABResult,
    SatisfactionStatus,
    VerdictType
)
from .core.exceptions import (
    CABEvaluationError,
    DockerValidationError,
    LLMError,
    InputTooLongError,
    AgentError,
    RepositoryError,
    ConfigurationError
)
from .agents.agent_factory import AgentFactory
from .workflows.generation_workflow import GenerationWorkflow
from .workflows.evaluation_workflow import EvaluationWorkflow
from .workflows.cab_workflow import CABWorkflow
from .utils.data_processor import DataProcessor
from .utils.repository_manager import RepositoryManager
from .utils.docker_manager import DockerManager
from .prompts.prompt_manager import PromptManager

__version__ = "1.0.0"

__all__ = [
    # Core
    'CABConfig',
    'DEFAULT_CONFIG',
    
    # Models
    'IssueData',
    'GenerationResult', 
    'EvaluationResult',
    'CABResult',
    'SatisfactionStatus',
    'VerdictType',
    
    # Exceptions
    'CABEvaluationError',
    'DockerValidationError',
    'LLMError',
    'InputTooLongError',
    'AgentError',
    'RepositoryError',
    'ConfigurationError',
    
    # Agents
    'AgentFactory',
    
    # Workflows
    'GenerationWorkflow',
    'EvaluationWorkflow', 
    'CABWorkflow',
    
    # Utilities
    'DataProcessor',
    'RepositoryManager',
    'DockerManager',
    'PromptManager'
]


def create_cab_evaluator(
    config_path: str = None,
    agent_model_mapping: dict = None,
    agent_framework_mapping: dict = None
) -> CABWorkflow:
    """Create a CAB evaluator with optional configuration.
    
    Args:
        config_path: Path to configuration file
        agent_model_mapping: Optional mapping of agents to models
        agent_framework_mapping: Optional mapping of agents to frameworks (e.g., {"maintainer": "openhands"})
        
    Returns:
        Configured CABWorkflow instance
    
    Example:
        >>> # Use default configuration
        >>> evaluator = create_cab_evaluator()
        
        >>> # Use custom model mapping
        >>> evaluator = create_cab_evaluator(
        ...     agent_model_mapping={
        ...         "maintainer": "sonnet37",
        ...         "user": "haiku", 
        ...         "judge": "sonnet"
        ...     }
        ... )
        
        >>> # Use OpenHands for maintainer agent
        >>> evaluator = create_cab_evaluator(
        ...     agent_model_mapping={
        ...         "maintainer": "claude-3.5-sonnet",
        ...         "user": "haiku",
        ...         "judge": "haiku"
        ...     },
        ...     agent_framework_mapping={
        ...         "maintainer": "openhands"
        ...     }
        ... )
    """
    if config_path:
        config = CABConfig.from_file(config_path)
    else:
        config = CABConfig()
    
    # Apply agent framework mapping if provided
    if agent_framework_mapping:
        if "maintainer" in agent_framework_mapping:
            config.agent_framework.maintainer_framework = agent_framework_mapping["maintainer"]
        # Note: Currently only maintainer framework is configurable
        # user and judge agents always use Strands framework
    
    # Validate and setup
    config.validate()
    config.setup_directories()
    
    return CABWorkflow(config)
