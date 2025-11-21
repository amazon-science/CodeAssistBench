"""Utility functions for Q-CLI integration."""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_qcli_installation(qcli_path: str = "q") -> bool:
    """Check if Q-CLI is installed and accessible.
    
    Args:
        qcli_path: Path to Q-CLI executable
        
    Returns:
        True if Q-CLI is available, False otherwise
    """
    try:
        result = subprocess.run(
            [qcli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info(f"✅ Q-CLI found: {result.stdout.strip()}")
            return True
        else:
            logger.warning(f"Q-CLI not working: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.warning(f"Q-CLI not found at: {qcli_path}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Q-CLI validation timeout")
        return False
    except Exception as e:
        logger.warning(f"Q-CLI validation error: {e}")
        return False


def get_qcli_version(qcli_path: str = "q") -> Optional[str]:
    """Get Q-CLI version.
    
    Args:
        qcli_path: Path to Q-CLI executable
        
    Returns:
        Version string or None if unavailable
    """
    try:
        result = subprocess.run(
            [qcli_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def get_available_models(qcli_path: str = "q") -> list:
    """Get list of available Q-CLI models.
    
    Args:
        qcli_path: Path to Q-CLI executable
        
    Returns:
        List of available model names
    """
    try:
        # Trigger error with invalid model to get available models list
        result = subprocess.run(
            [qcli_path, "chat", "--model", "invalid-model-name", "--no-interactive", "test"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Parse error message for available models
        if "Available models:" in result.stderr:
            models_str = result.stderr.split("Available models:")[1].strip()
            models = [m.strip() for m in models_str.split(",")]
            return models
        
        return []
    except Exception as e:
        logger.debug(f"Could not retrieve available models: {e}")
        return []
