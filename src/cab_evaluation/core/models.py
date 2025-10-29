"""Data models for CAB evaluation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class SatisfactionStatus(str, Enum):
    """User satisfaction status levels."""
    FULLY_SATISFIED = "FULLY_SATISFIED"
    PARTIALLY_SATISFIED = "PARTIALLY_SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"


class VerdictType(str, Enum):
    """Evaluation verdict types."""
    CORRECT = "CORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    INCORRECT = "INCORRECT"
    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"


class VerbosityLevel(str, Enum):
    """Response verbosity levels."""
    CONCISE = "CONCISE"
    APPROPRIATE = "APPROPRIATE"
    VERBOSE = "VERBOSE"


@dataclass
class CommitInfo:
    """Information about repository commit."""
    repository: str
    sha: str
    message: str
    author: str
    date: str


@dataclass
class Question:
    """Represents a user question."""
    title: str
    body: str
    user: str
    created_at: str


@dataclass
class Comment:
    """Represents a comment in the conversation."""
    user: str
    body: str
    created_at: str


@dataclass
class ConversationMessage:
    """Represents a message in agent conversation."""
    role: str  # "user", "maintainer", "judge"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IssueData:
    """Complete issue data structure."""
    id: str
    language: str
    commit_info: CommitInfo
    first_question: Question
    comments: List[Comment]
    user_satisfaction_condition: List[str]
    dockerfile: Optional[str] = None
    extra_files: Optional[Dict[str, str]] = None


@dataclass
class AlignmentCondition:
    """Individual alignment condition."""
    number: int
    satisfied: bool
    description: str


@dataclass
class AlignmentScore:
    """User satisfaction alignment scoring."""
    satisfied: int
    total: int
    percentage: float
    conditions: List[AlignmentCondition] = field(default_factory=list)
    technical_correctness: str = "UNKNOWN"
    verbosity: str = "UNKNOWN"


@dataclass
class DockerValidationResult:
    """Docker validation results."""
    success: bool
    logs: str
    test_commands: List[str] = field(default_factory=list)
    extra_files: Dict[str, str] = field(default_factory=dict)
    modified_dockerfile: bool = False
    error: Optional[str] = None


@dataclass
class LLMCallStats:
    """Statistics for LLM calls."""
    total_calls: int
    calls_by_agent: Dict[str, int] = field(default_factory=dict)
    calls_by_model: Dict[str, int] = field(default_factory=dict)


@dataclass
class ExplorationResult:
    """Results from repository exploration."""
    commands_executed: List[str]
    results: str
    exploration_log: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class RepositoryFile:
    """Information about a repository file."""
    path: str
    content: str
    size: int
    relevance_score: float = 0.0
    last_modified: Optional[str] = None
    file_type: str = ""


@dataclass
class RepositoryExploration:
    """Results from repository exploration."""
    files_explored: List[RepositoryFile] = field(default_factory=list)
    directory_structure: Dict[str, Any] = field(default_factory=dict)
    relevant_files: List[str] = field(default_factory=list)
    exploration_time_seconds: float = 0.0
    total_files_found: int = 0
    files_read: int = 0
    exploration_log: str = ""


@dataclass
class ConversationAnalysis:
    """Deep analysis of conversation history."""
    total_messages: int
    messages_by_role: Dict[str, int] = field(default_factory=dict)
    technical_solutions: List[str] = field(default_factory=list)
    solution_patterns: List[str] = field(default_factory=list)
    code_snippets: List[str] = field(default_factory=list)
    dependency_mentions: List[str] = field(default_factory=list)
    file_references: List[str] = field(default_factory=list)
    solution_evolution: List[str] = field(default_factory=list)
    analysis_summary: str = ""


@dataclass
class JudgeIteration:
    """Results from a single judge iteration."""
    iteration_number: int
    reasoning: str
    verdict: VerdictType
    confidence_score: float
    alignment_score: Optional[AlignmentScore] = None
    key_issues: List[str] = field(default_factory=list)
    files_examined: List[str] = field(default_factory=list)
    new_findings: List[str] = field(default_factory=list)
    iteration_time_seconds: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class IterativeEvaluationResult:
    """Results from iterative judge evaluation."""
    iterations: List[JudgeIteration] = field(default_factory=list)
    final_judgment: str = ""
    final_verdict: VerdictType = VerdictType.UNKNOWN
    final_alignment_score: Optional[AlignmentScore] = None
    final_key_issues: List[str] = field(default_factory=list)
    repository_exploration: Optional[RepositoryExploration] = None
    conversation_analysis: Dict[str, Any] = field(default_factory=dict)
    total_evaluation_time_seconds: float = 0.0
    stopped_early: bool = False
    early_stopping_reason: str = ""
    confidence_progression: List[float] = field(default_factory=list)
    total_token_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Results from generation workflow."""
    issue_data: IssueData
    modified_dockerfile: Optional[str] = None
    total_conversation_rounds: int = 0
    original_comment_count: int = 0
    user_satisfied: bool = False
    exploration_history: List[str] = field(default_factory=list)
    exploration_log: str = ""
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    llm_call_counter: Dict[str, int] = field(default_factory=dict)
    satisfaction_status: SatisfactionStatus = SatisfactionStatus.NOT_SATISFIED
    satisfaction_reason: str = ""
    final_answer: str = ""  # Final maintainer answer extracted from conversation
    prompt_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Cache metrics per agent


@dataclass
class EvaluationResult:
    """Results from evaluation workflow."""
    judgment: str
    verdict: VerdictType
    key_issues: List[str] = field(default_factory=list)
    alignment_score: Optional[AlignmentScore] = None
    docker_results: Optional[DockerValidationResult] = None
    llm_calls: Dict[str, int] = field(default_factory=dict)
    initial_alignment_score: Optional[AlignmentScore] = None
    final_alignment_score: Optional[AlignmentScore] = None
    prompt_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Cache metrics per agent
    
    # New fields for iterative evaluation
    iterative_evaluation: Optional[IterativeEvaluationResult] = None
    is_iterative: bool = False
    repository_path: Optional[str] = None


@dataclass
class CABResult:
    """Complete CAB workflow results."""
    issue_id: str
    question_title: str
    question_body: str
    generation_result: GenerationResult
    evaluation_result: EvaluationResult
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    max_conversation_rounds: int = 10
    max_exploration_iterations: int = 5
    command_timeout: int = 300
    overall_timeout: int = 600
    docker_build_timeout: int = 900
    docker_run_timeout: int = 600
    enable_docker_validation: bool = True
    batch_size: int = 10
    max_retries: int = 1000


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    model_name: str = "sonnet"
    temperature: float = 0.0
    max_tokens: int = 120000
    enable_thinking: bool = False
    system_prompt_path: Optional[str] = None


@dataclass
class JudgeConfig:
    """Configuration for judge agent behavior."""
    max_iterations: int = 10
    enable_repository_exploration: bool = True
    enable_conversation_analysis: bool = True
    exploration_file_limit: int = 20
    confidence_threshold: float = 0.85
    iteration_timeout_seconds: int = 300
    min_iterations: int = 1
    early_stopping_enabled: bool = True
