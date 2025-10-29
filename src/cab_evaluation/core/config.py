"""Configuration management for CAB evaluation."""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str
    model_id: str
    max_tokens: int
    temperature: float = 0.0
    region: str = "us-east-1"
    provider: str = "bedrock"
    api_key_env_var: Optional[str] = None
    thinking_enabled: bool = False


@dataclass
class DockerConfig:
    """Configuration for Docker operations."""
    build_timeout: int = 900
    run_timeout: int = 600
    max_retries: int = 3
    cleanup_after_run: bool = True
    enable_validation: bool = True


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    max_conversation_rounds: int = 10
    max_exploration_iterations: int = 5
    command_timeout: int = 300
    overall_timeout: int = 600
    batch_size: int = 10
    enable_parallel_processing: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_path: Optional[str] = None
    enable_llm_logging: bool = True
    llm_log_path: Optional[str] = None


@dataclass
class CABConfig:
    """Main configuration class for CAB evaluation."""
    
    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # Workflow configuration
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    
    # Docker configuration
    docker: DockerConfig = field(default_factory=DockerConfig)
    
    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paths
    prompts_dir: str = "prompts"
    results_dir: str = "results"
    temp_dir: str = "/tmp/cab_evaluation"
    
    # Agent settings - Updated to use sonnet37 as default for Strands framework
    default_maintainer_model: str = "sonnet37"
    default_user_model: str = "sonnet37"
    default_judge_model: str = "sonnet37"
    
    def __post_init__(self):
        """Initialize default model configurations."""
        if not self.models:
            self._setup_default_models()
    
    def _setup_default_models(self):
        """Setup default model configurations."""
        self.models = {
            "haiku": ModelConfig(
                name="haiku",
                model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
                max_tokens=120000,
                provider="bedrock"
            ),
            "sonnet": ModelConfig(
                name="sonnet",
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                max_tokens=120000,
                provider="bedrock"
            ),
            "sonnet37": ModelConfig(
                name="sonnet37",
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                max_tokens=120000,
                provider="bedrock"
            ),
            "thinking": ModelConfig(
                name="thinking",
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                max_tokens=120000,
                provider="bedrock",
                thinking_enabled=True
            ),
            "deepseek": ModelConfig(
                name="deepseek",
                model_id="us.deepseek.r1-v1:0",
                max_tokens=30000,
                provider="bedrock"
            ),
            "llama": ModelConfig(
                name="llama",
                model_id="us.meta.llama3-3-70b-instruct-v1:0",
                max_tokens=8192,
                provider="bedrock"
            )
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> 'CABConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return cls()
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading config from {config_path}: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CABConfig':
        """Create configuration from dictionary."""
        try:
            # Create nested objects
            models = {}
            if 'models' in config_dict:
                for name, model_data in config_dict['models'].items():
                    models[name] = ModelConfig(**model_data)
            
            workflow_config = WorkflowConfig()
            if 'workflow' in config_dict:
                workflow_config = WorkflowConfig(**config_dict['workflow'])
            
            docker_config = DockerConfig()
            if 'docker' in config_dict:
                docker_config = DockerConfig(**config_dict['docker'])
            
            logging_config = LoggingConfig()
            if 'logging' in config_dict:
                logging_config = LoggingConfig(**config_dict['logging'])
            
            # Remove nested configs from main dict to avoid duplicate field errors
            main_config = {k: v for k, v in config_dict.items() 
                          if k not in ['models', 'workflow', 'docker', 'logging']}
            
            return cls(
                models=models,
                workflow=workflow_config,
                docker=docker_config,
                logging=logging_config,
                **main_config
            )
        except Exception as e:
            raise ConfigurationError(f"Error creating config from dictionary: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        try:
            # Create directory if it doesn't exist
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Error saving config to {config_path}: {e}")
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration by name."""
        if model_name not in self.models:
            raise ConfigurationError(f"Unknown model: {model_name}")
        return self.models[model_name]
    
    def validate(self):
        """Validate configuration."""
        errors = []
        warnings = []
        
        # Check required directories exist or can be created
        for dir_path in [self.prompts_dir, self.results_dir]:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")
        
        # Validate model configurations
        bedrock_models = 0
        openai_models = 0
        
        for name, model_config in self.models.items():
            if model_config.provider == "openai":
                openai_models += 1
                if model_config.api_key_env_var:
                    if not os.getenv(model_config.api_key_env_var):
                        errors.append(f"Missing environment variable {model_config.api_key_env_var} for model {name}")
                else:
                    errors.append(f"OpenAI model {name} missing api_key_env_var configuration")
            elif model_config.provider == "bedrock":
                bedrock_models += 1
            else:
                warnings.append(f"Unknown provider '{model_config.provider}' for model {name}")
        
        # Check that we have at least one working model
        if bedrock_models == 0 and openai_models == 0:
            errors.append("No valid models configured")
        elif bedrock_models == 0 and openai_models > 0:
            warnings.append("Only OpenAI models configured - ensure GPT_TOKEN environment variable is set")
        
        # Validate default models exist
        for default_model in [self.default_maintainer_model, self.default_user_model, self.default_judge_model]:
            if default_model not in self.models:
                errors.append(f"Default model '{default_model}' not found in configured models")
        
        # Validate workflow settings
        if self.workflow.max_conversation_rounds <= 0:
            errors.append("max_conversation_rounds must be positive")
        
        if self.workflow.max_exploration_iterations <= 0:
            errors.append("max_exploration_iterations must be positive")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        logger.info(f"Configuration validation passed - {bedrock_models} Bedrock models, {openai_models} OpenAI models configured")
        if warnings:
            logger.info(f"Configuration warnings: {len(warnings)} warnings logged")
        return True
    
    def setup_directories(self):
        """Create required directories."""
        directories = [
            self.prompts_dir,
            self.results_dir,
            self.temp_dir,
            f"{self.prompts_dir}/maintainer",
            f"{self.prompts_dir}/user",
            f"{self.prompts_dir}/judge",
        ]
        
        for dir_path in directories:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            except Exception as e:
                raise ConfigurationError(f"Failed to create directory {dir_path}: {e}")


# Default configuration instance
DEFAULT_CONFIG = CABConfig()
