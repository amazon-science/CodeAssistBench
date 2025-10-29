"""Tests for CAB evaluation workflows."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from cab_evaluation.core.models import (
    IssueData,
    CommitInfo,
    Question,
    Comment,
    GenerationResult,
    SatisfactionStatus,
    VerdictType
)
from cab_evaluation.workflows import (
    GenerationWorkflow,
    EvaluationWorkflow,
    CABWorkflow
)
from cab_evaluation.core.config import CABConfig


@pytest.fixture
def sample_issue_data():
    """Create sample issue data for testing."""
    return IssueData(
        id="test_issue_1",
        language="python",
        commit_info=CommitInfo(
            repository="https://github.com/test/repo",
            sha="abc123def456",
            message="Test commit",
            author="test_user",
            date="2024-01-01T00:00:00Z"
        ),
        first_question=Question(
            title="Test issue",
            body="This is a test issue description",
            user="test_user",
            created_at="2024-01-01T00:00:00Z"
        ),
        comments=[
            Comment(
                user="maintainer",
                body="This is a test response",
                created_at="2024-01-01T01:00:00Z"
            )
        ],
        user_satisfaction_condition=[
            "Solution should explain the problem",
            "Solution should provide working code"
        ]
    )


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = CABConfig()
    # Override timeouts for faster tests
    config.workflow.command_timeout = 10
    config.workflow.overall_timeout = 60
    config.docker.build_timeout = 30
    config.docker.run_timeout = 30
    return config


class TestGenerationWorkflow:
    """Test generation workflow functionality."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_config):
        """Test workflow initialization."""
        workflow = GenerationWorkflow(mock_config)
        
        assert workflow.config == mock_config
        assert workflow.agent_factory is not None
        assert workflow.repository_manager is not None
        assert workflow.docker_manager is not None
    
    @pytest.mark.asyncio
    @patch('cab_evaluation.utils.repository_manager.RepositoryManager.clone_repository')
    @patch('cab_evaluation.agents.maintainer_agent.MaintainerAgent.choose_commit')
    async def test_run_generation_basic_flow(
        self,
        mock_choose_commit,
        mock_clone_repository,
        sample_issue_data,
        mock_config
    ):
        """Test basic generation workflow execution."""
        # Setup mocks
        mock_clone_repository.return_value = "/tmp/test_repo"
        mock_choose_commit.return_value = "abc123def456"
        
        workflow = GenerationWorkflow(mock_config)
        
        # Mock the interactive exploration method to avoid actual LLM calls
        async def mock_interactive_exploration(*args, **kwargs):
            return "Mock answer", ["Mock exploration"], "Mock exploration log"
        
        workflow._interactive_exploration = mock_interactive_exploration
        
        # Mock the conversation method
        async def mock_conduct_conversation(*args, **kwargs):
            return (
                [sample_issue_data.first_question, {"role": "maintainer", "content": "Mock response"}],
                1,
                {"satisfaction_status": SatisfactionStatus.FULLY_SATISFIED, "satisfaction_reason": "Test"}
            )
        
        workflow._conduct_conversation = mock_conduct_conversation
        
        # This test would require more extensive mocking for full functionality
        # For now, just test initialization
        assert workflow is not None


class TestEvaluationWorkflow:
    """Test evaluation workflow functionality."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_config):
        """Test evaluation workflow initialization."""
        workflow = EvaluationWorkflow(mock_config)
        
        assert workflow.config == mock_config
        assert workflow.agent_factory is not None
        assert workflow.docker_manager is not None
    
    def test_extract_final_maintainer_answer(self, mock_config):
        """Test extracting final maintainer answer."""
        workflow = EvaluationWorkflow(mock_config)
        
        # Test with ConversationMessage objects
        conversation = [
            Mock(role="user", content="User message"),
            Mock(role="maintainer", content="First maintainer response"),
            Mock(role="user", content="User follow-up"),
            Mock(role="maintainer", content="Final maintainer response")
        ]
        
        final_answer = workflow._extract_final_maintainer_answer(conversation)
        assert final_answer == "Final maintainer response"
        
        # Test with dictionary format
        dict_conversation = [
            {"role": "user", "content": "User message"},
            {"role": "maintainer", "content": "Final dict response"}
        ]
        
        final_answer = workflow._extract_final_maintainer_answer(dict_conversation)
        assert final_answer == "Final dict response"


class TestCABWorkflow:
    """Test complete CAB workflow functionality."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_config):
        """Test CAB workflow initialization."""
        workflow = CABWorkflow(mock_config)
        
        assert workflow.config == mock_config
        assert workflow.generation_workflow is not None
        assert workflow.evaluation_workflow is not None
        assert workflow.data_processor is not None
    
    def test_cab_result_to_dict(self, sample_issue_data, mock_config):
        """Test converting CAB result to dictionary."""
        workflow = CABWorkflow(mock_config)
        
        # Create mock CAB result
        from cab_evaluation.core.models import CABResult, GenerationResult, EvaluationResult, ConversationMessage
        
        generation_result = GenerationResult(
            issue_data=sample_issue_data,
            total_conversation_rounds=2,
            original_comment_count=1,
            user_satisfied=True,
            conversation_history=[
                ConversationMessage(role="user", content="Test user message"),
                ConversationMessage(role="maintainer", content="Test maintainer response")
            ],
            satisfaction_status=SatisfactionStatus.FULLY_SATISFIED
        )
        
        evaluation_result = EvaluationResult(
            judgment="Test judgment",
            verdict=VerdictType.CORRECT,
            key_issues=[],
            llm_calls={"judge": 1}
        )
        
        cab_result = CABResult(
            issue_id="test_issue",
            question_title="Test title",
            question_body="Test body",
            generation_result=generation_result,
            evaluation_result=evaluation_result
        )
        
        result_dict = workflow._cab_result_to_dict(cab_result)
        
        assert result_dict["issue_id"] == "test_issue"
        assert result_dict["question_title"] == "Test title"
        assert result_dict["final_verdict"] == "CORRECT"
        assert result_dict["user_satisfied"] == True
        assert result_dict["total_conversation_rounds"] == 2


@pytest.mark.asyncio
async def test_create_cab_evaluator():
    """Test the create_cab_evaluator function."""
    from cab_evaluation import create_cab_evaluator
    
    # Test with default config
    evaluator = create_cab_evaluator()
    assert evaluator is not None
    assert isinstance(evaluator, CABWorkflow)
    
    # Test with custom agent mapping
    custom_evaluator = create_cab_evaluator(
        agent_model_mapping={
            "maintainer": "haiku",
            "user": "sonnet", 
            "judge": "sonnet"
        }
    )
    assert custom_evaluator is not None
