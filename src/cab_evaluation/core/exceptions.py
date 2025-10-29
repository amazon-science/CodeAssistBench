"""Custom exceptions for CAB evaluation."""


class CABEvaluationError(Exception):
    """Base exception class for CAB evaluation errors."""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class DockerValidationError(CABEvaluationError):
    """Exception raised when Docker validation fails."""
    
    def __init__(self, message: str, docker_logs: str = None):
        self.docker_logs = docker_logs
        super().__init__(message, "DOCKER_ERROR")


class LLMError(CABEvaluationError):
    """Exception raised when LLM operations fail."""
    
    def __init__(self, message: str, model_name: str = None, retry_count: int = 0):
        self.model_name = model_name
        self.retry_count = retry_count
        super().__init__(message, "LLM_ERROR")


class InputTooLongError(LLMError):
    """Exception raised when input exceeds model context window."""
    
    def __init__(self, message: str = "Input too long for model context window"):
        super().__init__(message, error_code="INPUT_TOO_LONG")


class AgentError(CABEvaluationError):
    """Exception raised when agent operations fail."""
    
    def __init__(self, message: str, agent_type: str = None):
        self.agent_type = agent_type
        super().__init__(message, "AGENT_ERROR")


class RepositoryError(CABEvaluationError):
    """Exception raised when repository operations fail."""
    
    def __init__(self, message: str, repo_url: str = None):
        self.repo_url = repo_url
        super().__init__(message, "REPOSITORY_ERROR")


class ConfigurationError(CABEvaluationError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message, "CONFIG_ERROR")
