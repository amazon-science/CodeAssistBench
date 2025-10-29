"""Task prompts and constants."""

from typing import Dict


class TaskPrompts:
    """Constants for task-specific prompts."""
    
    # Docker-related prompts
    DOCKER_EXPLORATION = """
    This is a Docker-related issue. You can:
    1. Explore the repository
    2. Suggest modifications to the Dockerfile
    3. Provide Docker commands to solve the issue
    4. Create or modify additional files needed for Docker builds
    
    Format file creation/modifications as:
    CREATE_FILE[filename]:
    file content
    END_FILE
    
    Format Dockerfile modifications as:
    MODIFY_DOCKERFILE:
    # modified content
    END_DOCKERFILE
    """
    
    # Test command generation
    TEST_COMMAND_GENERATION = """
    You are helping to generate test commands to verify the solution to a user's Docker issue.
    
    Based on the user's issue description, the maintainer's response, and the exploration results,
    generate a series of commands that will verify if the solution works as expected.
    
    IMPORTANT: Generate commands that will run INSIDE the Docker container.
    DO NOT include 'docker' commands as these will be executed inside the container.
    
    Good commands:
    - Command-line tests that verify the application works
    - File existence checks (e.g., `test -f /path/to/file` but ONLY for files you know exist)
    - Service checks (e.g., `curl localhost:8080` but only if the container exposes this port)
    - Configuration validation
    
    Format your response as a JSON array of commands.
    """
    
    # Exploration prompts
    INITIAL_EXPLORATION = """
    First, assess the question and determine what files or code areas would be most relevant to explore.
    Respond with specific exploration commands that should be run to gather information.
    Format your response with exploration commands clearly labeled as:
    EXPLORE: <command to run>
    """
    
    CONTINUED_EXPLORATION = """
    Based on the information gathered so far, continue exploring the repository to better understand the issue.
    You can request additional files, search for specific patterns, or examine other areas of the codebase.
    Format your exploration commands clearly as:
    EXPLORE: <command to run>
    
    If you believe you have enough information to answer the question fully, begin your response with:
    ANSWER: <comprehensive answer to the user's question>
    """
    
    # Commit selection
    COMMIT_SELECTION = """
    You are a software maintainer helping with a code issue. You need to determine if the user has explicitly 
    mentioned a specific git commit hash they want you to examine.
    
    Your task:
    1. Carefully read the user's question
    2. Determine if they explicitly mentioned a specific commit hash they want you to look at
    3. If yes, extract and return ONLY that full commit hash as your response
    4. If no specific commit is mentioned, or if it's ambiguous, respond with ONLY "USE_REFERENCE_COMMIT"
    
    A git commit hash is typically a 40-character hexadecimal string (0-9, a-f), though users might mention 
    abbreviated versions (at least 7 characters).

    Only extract a hash if the user is clearly asking about a specific version/commit of the code.
    """
    
    # Final answer generation
    FINAL_ANSWER_GENERATION = """
    Based on all the exploration done so far, please provide a comprehensive answer to the user's question.
    Use all relevant information gathered during the exploration to formulate your response.
    """
    
    # Satisfaction evaluation
    SATISFACTION_EVALUATION = """
    After writing your response to the maintainer, add a separate section at the end that explicitly evaluates whether
    you are fully satisfied. Format this section as follows:
    
    SATISFACTION_STATUS: [FULLY_SATISFIED | PARTIALLY_SATISFIED | NOT_SATISFIED]
    REASON: <brief explanation of why you are or are not satisfied>
    
    This section will be removed before sending your response to the maintainer.
    """


class ResponseFormats:
    """Standard response format templates."""
    
    JUDGE_EVALUATION_FORMAT = """
    TECHNICAL CORRECTNESS: [CORRECT/PARTIALLY CORRECT/INCORRECT]
    
    ALIGNMENT SCORE: X/Y CONDITIONS MET (Z%)
    
    CONDITION 1: [TRUE/FALSE] <brief description of condition>
    CONDITION 2: [TRUE/FALSE] <brief description of condition>
    
    VERBOSITY ASSESSMENT: [CONCISE/APPROPRIATE/VERBOSE]
    
    VERDICT: [CORRECT/PARTIALLY CORRECT/INCORRECT]
    
    KEY ISSUES: List ALL issues with the maintainer's answer
    
    REASONING: Detailed explanation of your verdict
    """
    
    FILE_CREATION_FORMAT = """
    CREATE_FILE[filename]:
    file content
    END_FILE
    """
    
    DOCKERFILE_MODIFICATION_FORMAT = """
    MODIFY_DOCKERFILE:
    # modified content
    END_DOCKERFILE
    """


class ValidationPatterns:
    """Regex patterns for validation."""
    
    # Error patterns for detecting input too long errors
    INPUT_TOO_LONG_PATTERNS = [
        "input is too long",
        "input is t\noo long",
        "input exceeds maximum token length", 
        "context window exceeded",
        "input sequence length exceeds the model's context window",
        "input too long for model",
        "input is too long for requested model",
        "ValidationException) when calling the InvokeModel operation: Input is t\noo long",
        "ValidationException) when calling the InvokeModel operation: Input is too long",
        "too many total text bytes",
        "Member must have length less than or equal to",
        "body' failed to satisfy constraint",
        "failed to satisfy constraint: Member must have length"
    ]
    
    # Git commit hash patterns
    GIT_COMMIT_PATTERN = r'\b[0-9a-f]{7,40}\b'
    
    # File creation patterns
    CREATE_FILE_PATTERN = r'CREATE_FILE\[(.+?)\]:(.*?)END_FILE'
    MODIFY_DOCKERFILE_PATTERN = r'MODIFY_DOCKERFILE:(.*?)END_DOCKERFILE'


# Model mapping for backward compatibility
MODEL_MAPPING = {
    "haiku": "haiku",
    "sonnet": "sonnet", 
    "sonnet37": "sonnet37",
    "thinking": "thinking",
    "deepseek": "deepseek",
    "deepseek-r1": "deepseek",
    "llama": "llama"
}
