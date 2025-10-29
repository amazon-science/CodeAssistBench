"""Tests for CAB configuration."""

import json
import pytest
import tempfile
from pathlib import Path

from cab_evaluation.core.config import CABConfig, ModelConfig
from cab_evaluation.core.exceptions import ConfigurationError


class TestCABConfig:
    """Test CAB configuration functionality."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = CABConfig()
        
        assert config.default_maintainer_model == "haiku"
        assert config.default_user_model == "sonnet"
        assert config.default_judge_model == "sonnet"
        assert len(config.models) > 0
        assert "haiku" in config.models
        assert "sonnet" in config.models
    
    def test_model_configuration(self):
        """Test model configuration."""
        config = CABConfig()
        
        # Test getting existing model
        haiku_config = config.get_model_config("haiku")
        assert haiku_config.name == "haiku"
        assert haiku_config.provider == "bedrock"
        assert haiku_config.max_tokens == 120000
        
        # Test getting non-existent model
        with pytest.raises(ConfigurationError):
            config.get_model_config("nonexistent")
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "default_maintainer_model": "custom_model",
            "workflow": {
                "max_conversation_rounds": 15,
                "batch_size": 20
            },
            "docker": {
                "build_timeout": 1200
            }
        }
        
        config = CABConfig.from_dict(config_dict)
        
        assert config.default_maintainer_model == "custom_model"
        assert config.workflow.max_conversation_rounds == 15
        assert config.workflow.batch_size == 20
        assert config.docker.build_timeout == 1200
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = CABConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "models" in config_dict
        assert "workflow" in config_dict
        assert "docker" in config_dict
        assert "logging" in config_dict
    
    def test_config_file_operations(self):
        """Test saving and loading config files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Create and save config
            original_config = CABConfig()
            original_config.default_maintainer_model = "test_model"
            original_config.save_to_file(config_path)
            
            # Load config from file
            loaded_config = CABConfig.from_file(config_path)
            
            assert loaded_config.default_maintainer_model == "test_model"
            
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = CABConfig()
        
        # Valid configuration should pass
        assert config.validate() == True
        
        # Invalid configuration should fail
        config.workflow.max_conversation_rounds = -1
        with pytest.raises(ConfigurationError):
            config.validate()


class TestModelConfig:
    """Test model configuration."""
    
    def test_model_config_creation(self):
        """Test creating model configuration."""
        model_config = ModelConfig(
            name="test_model",
            model_id="test.model.id",
            max_tokens=50000,
            provider="bedrock"
        )
        
        assert model_config.name == "test_model"
        assert model_config.model_id == "test.model.id"
        assert model_config.max_tokens == 50000
        assert model_config.provider == "bedrock"
        assert model_config.temperature == 0.0  # default
