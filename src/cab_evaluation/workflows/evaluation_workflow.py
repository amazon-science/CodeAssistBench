"""Evaluation workflow for CAB evaluation."""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..core.models import (
    GenerationResult,
    EvaluationResult, 
    IssueData,
    VerdictType,
    DockerValidationResult,
    JudgeConfig,
    ConversationMessage,
    IterativeEvaluationResult
)
from ..core.config import CABConfig
from ..core.exceptions import CABEvaluationError
from ..agents.agent_factory import AgentFactory
from ..utils.docker_manager import DockerManager

logger = logging.getLogger(__name__)


class EvaluationWorkflow:
    """Handles the evaluation workflow - judges final maintainer answer."""
    
    def __init__(self, config: Optional[CABConfig] = None, judge_config: Optional[JudgeConfig] = None):
        """Initialize evaluation workflow.
        
        Args:
            config: CAB configuration
            judge_config: Configuration for iterative judge behavior
        """
        self.config = config or CABConfig()
        self.judge_config = judge_config or JudgeConfig()
        self.agent_factory = AgentFactory(self.config)
        self.docker_manager = DockerManager(self.config.docker)
        
    async def run_evaluation(
        self,
        generation_result: GenerationResult,
        agent_model_mapping: Optional[Dict[str, str]] = None
    ) -> EvaluationResult:
        """Run evaluation workflow on generation results.
        
        Args:
            generation_result: Results from generation workflow
            agent_model_mapping: Optional mapping of agent types to model names
            
        Returns:
            EvaluationResult with judgment and metrics
        """
        logger.info(f"Starting evaluation workflow for issue: {generation_result.issue_data.id}")
        
        # Create judge agent
        if agent_model_mapping and "judge" in agent_model_mapping:
            judge_agent = self.agent_factory.create_judge_agent(agent_model_mapping["judge"])
        else:
            judge_agent = self.agent_factory.create_judge_agent()
        
        # Get final maintainer answer - prefer stored final_answer, fallback to extraction
        if hasattr(generation_result, 'final_answer') and generation_result.final_answer:
            final_answer = generation_result.final_answer
            logger.info(f"Using stored final_answer from generation result ({len(final_answer)} chars)")
        else:
            final_answer = self._extract_final_maintainer_answer(generation_result.conversation_history)
            logger.info(f"Extracted final_answer from conversation history ({len(final_answer)} chars)")
        
        # Run final Docker validation if this is a Docker issue
        docker_results = None
        if generation_result.issue_data.dockerfile:
            logger.info("Running final Docker validation for evaluation...")
            docker_results = await self._run_final_docker_validation(
                generation_result.issue_data,
                final_answer,
                generation_result.exploration_log
            )
        
        # Judge the final answer
        logger.info("Judging final maintainer answer...")
        judgment, verdict, key_issues, alignment_score = await judge_agent.judge_maintainer_answer(
            generation_result.issue_data,
            final_answer,
            docker_results
        )
        
        # Get judge LLM call statistics
        judge_llm_calls = judge_agent.get_call_statistics(generation_result.issue_data.id)
        
        # Combine LLM call statistics from generation and evaluation
        total_llm_calls = generation_result.llm_call_counter.copy()
        for agent_type, calls in judge_llm_calls.items():
            total_llm_calls[agent_type] = total_llm_calls.get(agent_type, 0) + calls
        
        # Collect prompt cache metrics from judge agent
        prompt_cache_metrics = {}
        
        # Copy existing cache metrics from generation result
        if hasattr(generation_result, 'prompt_cache') and generation_result.prompt_cache:
            prompt_cache_metrics = generation_result.prompt_cache.copy()
        
        # Get cache metrics from judge agent if it's a StrandsAgent
        if hasattr(judge_agent, '_calculate_cache_efficiency') and hasattr(judge_agent, '_strands_agent') and judge_agent._strands_agent:
            try:
                metrics_summary = judge_agent._strands_agent.event_loop_metrics.get_summary()
                usage = metrics_summary["accumulated_usage"]
                cache_efficiency = judge_agent._calculate_cache_efficiency(usage)
                
                prompt_cache_metrics["judge"] = {
                    "input_tokens": usage.get("inputTokens", 0),
                    "output_tokens": usage.get("outputTokens", 0),
                    "total_tokens": usage.get("totalTokens", 0),
                    "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
                    "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
                    "cache_hit_rate_percent": cache_efficiency.get("cache_hit_rate_percent", 0.0),
                    "cache_savings_usd": cache_efficiency.get("cache_savings_usd", 0.0)
                }
            except Exception as e:
                logger.warning(f"Failed to collect judge cache metrics: {e}")
        
        # Create Docker validation result object if available
        docker_validation_result = None
        if docker_results:
            docker_validation_result = DockerValidationResult(
                success=docker_results.get('success', False),
                logs=docker_results.get('logs', ''),
                test_commands=docker_results.get('test_commands', []),
                extra_files=docker_results.get('extra_files', {}),
                error=docker_results.get('error')
            )
        
        # Create evaluation result
        result = EvaluationResult(
            judgment=judgment,
            verdict=verdict,
            key_issues=key_issues,
            alignment_score=alignment_score,
            docker_results=docker_validation_result,
            llm_calls=total_llm_calls,
            final_alignment_score=alignment_score,  # In evaluation workflow, this is the only alignment score
            prompt_cache=prompt_cache_metrics
        )
        
        logger.info(f"Evaluation workflow complete for issue {generation_result.issue_data.id}")
        logger.info(f"Final verdict: {verdict.value}")
        if alignment_score:
            logger.info(f"Alignment score: {alignment_score.satisfied}/{alignment_score.total} conditions met ({alignment_score.percentage:.1f}%)")
        logger.info(f"Total LLM calls: {sum(total_llm_calls.values())}")
        
        return result

    async def run_iterative_evaluation(
        self,
        generation_result: GenerationResult,
        repository_path: str,
        agent_model_mapping: Optional[Dict[str, str]] = None
    ) -> EvaluationResult:
        """Run iterative evaluation workflow with repository exploration and multiple refinement iterations.
        
        Args:
            generation_result: Results from generation workflow
            repository_path: Path to repository for exploration
            agent_model_mapping: Optional mapping of agent types to model names
            
        Returns:
            EvaluationResult with iterative judgment and comprehensive metrics
        """
        logger.info(f"🚀 Starting ITERATIVE evaluation workflow for issue: {generation_result.issue_data.id}")
        logger.info(f"📁 Repository path: {repository_path}")
        logger.info(f"⚙️  Judge config: max_iterations={self.judge_config.max_iterations}, enable_repo_exploration={self.judge_config.enable_repository_exploration}")
        
        # Create judge agent with iterative configuration
        if agent_model_mapping and "judge" in agent_model_mapping:
            judge_agent = self.agent_factory.create_judge_agent(agent_model_mapping["judge"], judge_config=self.judge_config)
        else:
            judge_agent = self.agent_factory.create_judge_agent(judge_config=self.judge_config)
        
        # Get final maintainer answer
        if hasattr(generation_result, 'final_answer') and generation_result.final_answer:
            final_answer = generation_result.final_answer
            logger.info(f"Using stored final_answer from generation result ({len(final_answer)} chars)")
        else:
            final_answer = self._extract_final_maintainer_answer(generation_result.conversation_history)
            logger.info(f"Extracted final_answer from conversation history ({len(final_answer)} chars)")
        
        # Run final Docker validation if this is a Docker issue
        docker_results = None
        if generation_result.issue_data.dockerfile:
            logger.info("🐳 Running final Docker validation for evaluation...")
            docker_results = await self._run_final_docker_validation(
                generation_result.issue_data,
                final_answer,
                generation_result.exploration_log
            )
        
        # Convert conversation history to ConversationMessage format if needed
        conversation_messages = self._convert_conversation_history(generation_result.conversation_history)
        
        # Run iterative judgment
        logger.info("🎯 Starting iterative judge evaluation...")
        iterative_result = await judge_agent.judge_maintainer_answer_iterative(
            generation_result.issue_data,
            final_answer,
            repository_path,
            conversation_messages,
            docker_results
        )
        
        # Get judge LLM call statistics
        judge_llm_calls = judge_agent.get_call_statistics(generation_result.issue_data.id)
        
        # Combine LLM call statistics from generation and evaluation
        total_llm_calls = generation_result.llm_call_counter.copy()
        for agent_type, calls in judge_llm_calls.items():
            total_llm_calls[agent_type] = total_llm_calls.get(agent_type, 0) + calls
        
        # Add iterative evaluation token usage
        for agent_type, calls in iterative_result.total_token_usage.items():
            total_llm_calls[agent_type] = total_llm_calls.get(agent_type, 0) + calls
        
        # Collect prompt cache metrics from judge agent
        prompt_cache_metrics = {}
        
        # Copy existing cache metrics from generation result
        if hasattr(generation_result, 'prompt_cache') and generation_result.prompt_cache:
            prompt_cache_metrics = generation_result.prompt_cache.copy()
        
        # Get cache metrics from judge agent if it's a StrandsAgent
        if hasattr(judge_agent, '_calculate_cache_efficiency') and hasattr(judge_agent, '_strands_agent') and judge_agent._strands_agent:
            try:
                metrics_summary = judge_agent._strands_agent.event_loop_metrics.get_summary()
                usage = metrics_summary["accumulated_usage"]
                cache_efficiency = judge_agent._calculate_cache_efficiency(usage)
                
                prompt_cache_metrics["judge_iterative"] = {
                    "input_tokens": usage.get("inputTokens", 0),
                    "output_tokens": usage.get("outputTokens", 0),
                    "total_tokens": usage.get("totalTokens", 0),
                    "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
                    "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
                    "cache_hit_rate_percent": cache_efficiency.get("cache_hit_rate_percent", 0.0),
                    "cache_savings_usd": cache_efficiency.get("cache_savings_usd", 0.0)
                }
            except Exception as e:
                logger.warning(f"Failed to collect iterative judge cache metrics: {e}")
        
        # Create Docker validation result object if available
        docker_validation_result = None
        if docker_results:
            docker_validation_result = DockerValidationResult(
                success=docker_results.get('success', False),
                logs=docker_results.get('logs', ''),
                test_commands=docker_results.get('test_commands', []),
                extra_files=docker_results.get('extra_files', {}),
                error=docker_results.get('error')
            )
        
        # Create evaluation result with iterative data
        result = EvaluationResult(
            judgment=iterative_result.final_judgment,
            verdict=iterative_result.final_verdict,
            key_issues=iterative_result.final_key_issues,
            alignment_score=iterative_result.final_alignment_score,
            docker_results=docker_validation_result,
            llm_calls=total_llm_calls,
            final_alignment_score=iterative_result.final_alignment_score,
            prompt_cache=prompt_cache_metrics,
            # New iterative fields
            iterative_evaluation=iterative_result,
            is_iterative=True,
            repository_path=repository_path
        )
        
        # Log comprehensive results
        logger.info(f"🏁 Iterative evaluation workflow complete for issue {generation_result.issue_data.id}")
        logger.info(f"📊 Iterations completed: {len(iterative_result.iterations)}/{self.judge_config.max_iterations}")
        logger.info(f"⏱️  Total evaluation time: {iterative_result.total_evaluation_time_seconds:.2f}s")
        logger.info(f"🎯 Final verdict: {iterative_result.final_verdict.value}")
        if iterative_result.final_alignment_score:
            logger.info(f"📈 Final alignment: {iterative_result.final_alignment_score.satisfied}/{iterative_result.final_alignment_score.total} conditions ({iterative_result.final_alignment_score.percentage:.1f}%)")
        logger.info(f"🔧 Total LLM calls: {sum(total_llm_calls.values())}")
        if iterative_result.stopped_early:
            logger.info(f"⚡ Early stopping: {iterative_result.early_stopping_reason}")
        if iterative_result.repository_exploration:
            logger.info(f"📁 Repository exploration: {iterative_result.repository_exploration.files_read} files read in {iterative_result.repository_exploration.exploration_time_seconds:.2f}s")
        
        # Log confidence progression
        if iterative_result.confidence_progression:
            confidence_trend = " → ".join([f"{c:.2f}" for c in iterative_result.confidence_progression])
            logger.info(f"📊 Confidence progression: {confidence_trend}")
        
        return result

    def _convert_conversation_history(self, conversation_history: List[Any]) -> List[ConversationMessage]:
        """Convert conversation history to ConversationMessage format.
        
        Args:
            conversation_history: Raw conversation history
            
        Returns:
            List of ConversationMessage objects
        """
        messages = []
        
        for msg in conversation_history:
            if isinstance(msg, ConversationMessage):
                messages.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                messages.append(ConversationMessage(
                    role=msg.role,
                    content=msg.content,
                    timestamp=getattr(msg, 'timestamp', None),
                    metadata=getattr(msg, 'metadata', {})
                ))
            elif isinstance(msg, dict):
                messages.append(ConversationMessage(
                    role=msg.get('role', 'unknown'),
                    content=msg.get('content', ''),
                    metadata=msg.get('metadata', {})
                ))
            else:
                logger.warning(f"Unknown conversation message format: {type(msg)}")
        
        return messages
    
    def _extract_final_maintainer_answer(self, conversation_history) -> str:
        """Extract the final maintainer answer from conversation history.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Final maintainer answer text
        """
        # Find the last maintainer message
        for message in reversed(conversation_history):
            if hasattr(message, 'role'):
                if message.role == "maintainer":
                    return message.content
            elif isinstance(message, dict):
                if message.get('role') == "maintainer":
                    return message.get('content', '')
        
        # Fallback to first maintainer message if no later ones found
        for message in conversation_history:
            if hasattr(message, 'role'):
                if message.role == "maintainer":
                    return message.content
            elif isinstance(message, dict):
                if message.get('role') == "maintainer":
                    return message.get('content', '')
        
        logger.warning("No maintainer response found in conversation history")
        return "No maintainer response found"
    
    async def _run_final_docker_validation(
        self,
        issue_data: IssueData,
        final_answer: str,
        exploration_log: str
    ) -> Dict[str, Any]:
        """Run final Docker validation for evaluation.
        
        Args:
            issue_data: Issue data
            final_answer: Final maintainer answer
            exploration_log: Exploration log from generation
            
        Returns:
            Dictionary with Docker validation results
        """
        try:
            # Generate test commands based on final answer
            # For now, using empty list - could implement proper test command generation
            test_commands = []
            
            # Validate the Docker solution
            docker_result = self.docker_manager.validate_docker_solution(issue_data, test_commands)
            
            return {
                'success': docker_result.success,
                'logs': docker_result.logs,
                'test_commands': docker_result.test_commands,
                'extra_files': getattr(issue_data, 'extra_files', {}) or {},
                'error': docker_result.error
            }
            
        except Exception as e:
            logger.error(f"Error during final Docker validation: {e}")
            return {
                'success': False,
                'logs': f"Final Docker validation error: {str(e)}",
                'test_commands': [],
                'error': str(e)
            }
