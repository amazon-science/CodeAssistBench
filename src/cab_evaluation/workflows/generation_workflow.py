"""Generation workflow for CAB evaluation."""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple

from ..core.models import (
    IssueData, 
    GenerationResult, 
    ConversationMessage,
    SatisfactionStatus,
    ExplorationResult
)
from ..core.config import CABConfig
from ..core.exceptions import CABEvaluationError, InputTooLongError
from ..agents.agent_factory import AgentFactory
from ..utils.repository_manager import RepositoryManager, execute_command
from ..utils.docker_manager import DockerManager
from ..prompts.constants import TaskPrompts

logger = logging.getLogger(__name__)


class GenerationWorkflow:
    """Handles the generation workflow - conversation between maintainer and user agents."""
    
    def __init__(self, config: Optional[CABConfig] = None):
        """Initialize generation workflow.
        
        Args:
            config: CAB configuration
        """
        self.config = config or CABConfig()
        self.agent_factory = AgentFactory(self.config)
        self.repository_manager = RepositoryManager()
        self.docker_manager = DockerManager(self.config.docker)
        
    async def run_generation(
        self,
        issue_data: IssueData,
        agent_model_mapping: Optional[Dict[str, str]] = None
    ) -> GenerationResult:
        """Run generation workflow for an issue.
        
        Args:
            issue_data: Issue data to process
            agent_model_mapping: Optional mapping of agent types to model names
            
        Returns:
            GenerationResult with conversation and exploration data
        """
        logger.info(f"Starting generation workflow for issue: {issue_data.first_question.title}")
        
        # Create agents
        if agent_model_mapping:
            agents = self.agent_factory.update_model_mapping(agent_model_mapping)
        else:
            agents = self.agent_factory.create_agent_set()
        
        maintainer_agent = agents["maintainer"]
        user_agent = agents["user"]
        
        # Reset call counter for this issue
        maintainer_agent.reset_call_counter(issue_data.id)
        
        # Clone repository for exploration
        repo_url = self.repository_manager.parse_repo_name(issue_data.commit_info.repository)
        
        # Let maintainer choose commit
        question = f"{issue_data.first_question.title}\n\n{issue_data.first_question.body}"
        selected_commit = await maintainer_agent.choose_commit(
            issue_data.commit_info.sha, question
        )
        
        logger.info(f"Selected commit for exploration: {selected_commit}")
        
        # Clone repository
        repo_dir = self.repository_manager.clone_repository(repo_url, selected_commit)
        if not repo_dir:
            raise CABEvaluationError(f"Failed to clone repository: {repo_url}")
        
        try:
            # Perform interactive exploration
            logger.info("Starting interactive exploration")
            initial_answer, exploration_history, exploration_log = await self._interactive_exploration(
                repo_dir, question, maintainer_agent, issue_data.id
            )
            
            logger.info(f"Exploration complete. Initial answer length: {len(initial_answer)}")
            
            # Initialize conversation history
            conversation_history = [
                ConversationMessage(
                    role="user",
                    content=f"{issue_data.first_question.title}\n\n{issue_data.first_question.body}"
                ),
                ConversationMessage(role="maintainer", content=initial_answer)
            ]
            
            # Store original comment count
            original_comment_count = len(issue_data.comments)
            
            # Run Docker validation for initial answer if needed
            docker_results = None
            if issue_data.dockerfile:
                logger.info("Running initial Docker validation...")
                docker_results = await self._run_docker_validation(
                    issue_data, initial_answer, exploration_log
                )
            
            # Conduct conversation between agents
            logger.info("Starting agent conversation")
            final_conversation, total_rounds, final_satisfaction = await self._conduct_conversation(
                repo_dir, issue_data, conversation_history, maintainer_agent, user_agent, docker_results, exploration_log
            )
            
            # Get final LLM call statistics
            llm_call_stats = maintainer_agent.get_call_statistics(issue_data.id)
            
            # Extract final maintainer answer from conversation history
            final_answer = self._extract_final_maintainer_answer(final_conversation)
            
            # Collect prompt cache metrics from agents
            prompt_cache_metrics = {}
            
            # Get cache metrics from maintainer agent (StrandsAgent)
            if hasattr(maintainer_agent, '_calculate_cache_efficiency') and maintainer_agent._strands_agent:
                try:
                    metrics_summary = maintainer_agent._strands_agent.event_loop_metrics.get_summary()
                    usage = metrics_summary["accumulated_usage"]
                    cache_efficiency = maintainer_agent._calculate_cache_efficiency(usage)
                    
                    prompt_cache_metrics["maintainer"] = {
                        "input_tokens": usage.get("inputTokens", 0),
                        "output_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0),
                        "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
                        "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
                        "cache_hit_rate_percent": cache_efficiency.get("cache_hit_rate_percent", 0.0),
                        "cache_savings_usd": cache_efficiency.get("cache_savings_usd", 0.0)
                    }
                except Exception as e:
                    logger.warning(f"Failed to collect maintainer cache metrics: {e}")
            
            # Get cache metrics from user agent if it's also a StrandsAgent
            if hasattr(user_agent, '_calculate_cache_efficiency') and hasattr(user_agent, '_strands_agent') and user_agent._strands_agent:
                try:
                    metrics_summary = user_agent._strands_agent.event_loop_metrics.get_summary()
                    usage = metrics_summary["accumulated_usage"]
                    cache_efficiency = user_agent._calculate_cache_efficiency(usage)
                    
                    prompt_cache_metrics["user"] = {
                        "input_tokens": usage.get("inputTokens", 0),
                        "output_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0),
                        "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
                        "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
                        "cache_hit_rate_percent": cache_efficiency.get("cache_hit_rate_percent", 0.0),
                        "cache_savings_usd": cache_efficiency.get("cache_savings_usd", 0.0)
                    }
                except Exception as e:
                    logger.warning(f"Failed to collect user cache metrics: {e}")
            
            # Create result
            result = GenerationResult(
                issue_data=issue_data,
                modified_dockerfile=getattr(issue_data, 'modified_dockerfile', None),
                total_conversation_rounds=total_rounds,
                original_comment_count=original_comment_count,
                user_satisfied=final_satisfaction["satisfaction_status"] == SatisfactionStatus.FULLY_SATISFIED,
                exploration_history=exploration_history,
                exploration_log=exploration_log,
                conversation_history=final_conversation,
                llm_call_counter=llm_call_stats,
                satisfaction_status=final_satisfaction["satisfaction_status"],
                satisfaction_reason=final_satisfaction["satisfaction_reason"],
                final_answer=final_answer,
                prompt_cache=prompt_cache_metrics
            )
            
            logger.info(f"Generation workflow complete for issue {issue_data.id}")
            logger.info(f"User satisfied: {result.user_satisfied}")
            logger.info(f"Total conversation rounds: {total_rounds}")
            logger.info(f"Total LLM calls: {sum(llm_call_stats.values())}")
            
            return result
            
        finally:
            # Cleanup repository
            self.repository_manager.cleanup_repository(repo_dir)
    
    async def _interactive_exploration(
        self,
        repo_dir: str,
        question: str,
        maintainer_agent,
        issue_id: str,
        max_iterations: int = 5
    ) -> Tuple[str, List[str], str]:
        """Perform interactive repository exploration.
        
        Args:
            repo_dir: Repository directory
            question: User's question
            maintainer_agent: Maintainer agent instance
            issue_id: Issue ID for tracking
            max_iterations: Maximum exploration iterations
            
        Returns:
            Tuple of (final_answer, exploration_history, exploration_log)
        """
        exploration_history = []
        exploration_log = ""
        
        # Estimate context size limits (150K tokens ~= 600K chars)
        max_context_size = 600000
        current_exploration_context = ""
        
        logger.info(f"Starting interactive exploration with max {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            logger.info(f"Exploration iteration {iteration+1}/{max_iterations}")
            
            # Create system prompt based on iteration
            if iteration == 0:
                system_prompt = maintainer_agent.get_system_prompt() + TaskPrompts.INITIAL_EXPLORATION
                user_prompt = f"Question: {question}\n\nPlease help me understand this code issue."
            else:
                system_prompt = maintainer_agent.get_system_prompt() + TaskPrompts.CONTINUED_EXPLORATION
                user_prompt = f"Question: {question}\n\nExploration results so far:\n{current_exploration_context}\n\nPlease continue exploring or provide an answer."
            
            # Get exploration plan from maintainer
            try:
                exploration_plan = await maintainer_agent.call_llm(
                    user_prompt, system_prompt, issue_id
                )
                
                # Check for input too long error
                if exploration_plan == "ERROR_INPUT_TOO_LONG":
                    logger.warning("Input too long error. Stopping exploration.")
                    exploration_log += "\n--- EXPLORATION STOPPED: Input too long error ---\n"
                    break
                
                exploration_history.append(exploration_plan)
                logger.info(f"Received exploration plan ({len(exploration_plan)} chars)")
                
            except InputTooLongError:
                logger.warning("Input too long error in exploration. Stopping exploration.")
                exploration_log += "\n--- EXPLORATION STOPPED: Input too long error ---\n"
                break
            except Exception as e:
                logger.error(f"Error getting exploration plan: {e}")
                exploration_log += f"\n--- ERROR IN ITERATION {iteration+1} ---\n{str(e)}\n"
                break
            
            # Extract and execute exploration commands
            iteration_results = ""
            if "EXPLORE:" in exploration_plan:
                commands = [
                    line.split("EXPLORE: ", 1)[1].strip() 
                    for line in exploration_plan.split('\n') 
                    if line.strip().startswith("EXPLORE:")
                ]
                
                logger.info(f"Executing {len(commands)} exploration commands")
                
                for i, cmd in enumerate(commands):
                    try:
                        logger.info(f"Executing command {i+1}/{len(commands)}: {cmd}")
                        result = execute_command(repo_dir, cmd, timeout=self.config.workflow.command_timeout)
                        iteration_results += f"Command: {cmd}\nResult:\n{result}\n\n"
                    except Exception as e:
                        error_msg = f"Error executing command: {cmd}\nError: {str(e)}\n\n"
                        logger.error(f"Command execution error: {str(e)}")
                        iteration_results += error_msg
            
            # Add iteration results to full log
            exploration_log += f"\n--- ITERATION {iteration+1} ---\n{iteration_results}"
            
            # Check context size before adding to current context
            new_context = current_exploration_context + f"\n--- ITERATION {iteration+1} ---\n{iteration_results}"
            
            if len(new_context) > max_context_size:
                logger.warning(f"Exploration context would exceed size limit. Stopping exploration.")
                exploration_log += "\n--- EXPLORATION STOPPED: Context size limit ---\n"
                break
            else:
                current_exploration_context = new_context
            
            # Check for answer
            if "ANSWER:" in exploration_plan:
                logger.info("Found ANSWER section. Extracting final answer.")
                answer_part = exploration_plan.split("ANSWER:", 1)[1].strip()
                return answer_part, exploration_history, exploration_log
        
        # Generate final answer if no explicit answer found
        logger.info("Generating final answer from exploration results")
        final_system_prompt = maintainer_agent.get_system_prompt() + TaskPrompts.FINAL_ANSWER_GENERATION
        final_user_prompt = f"""
        Question: {question}
        
        Exploration results:
        {current_exploration_context}
        
        Please provide a comprehensive answer based on the exploration results above.
        """
        
        try:
            final_answer = await maintainer_agent.call_llm(
                final_user_prompt, final_system_prompt, issue_id
            )
            
            if final_answer == "ERROR_INPUT_TOO_LONG":
                logger.warning("Input too long in final answer generation. Using fallback.")
                final_answer = "After extensive repository exploration, I encountered context limitations. Based on the exploration conducted, I can provide relevant information about this issue."
            
            return final_answer, exploration_history, exploration_log
            
        except InputTooLongError:
            logger.warning("Input too long in final answer generation. Using fallback.")
            final_answer = "After extensive repository exploration, I encountered context limitations. Based on the exploration conducted, I can provide relevant information about this issue."
            return final_answer, exploration_history, exploration_log
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            fallback_answer = f"Based on the exploration conducted, I can provide information about this issue. Note: Full analysis was interrupted due to error: {str(e)}"
            return fallback_answer, exploration_history, exploration_log
    
    async def _conduct_conversation(
        self,
        repo_dir: str,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        maintainer_agent,
        user_agent,
        initial_docker_results: Optional[Dict[str, Any]] = None,
        exploration_log: str = ""
    ) -> Tuple[List[ConversationMessage], int, Dict[str, Any]]:
        """Conduct conversation between user and maintainer agents.
        
        Args:
            repo_dir: Repository directory
            issue_data: Issue data
            conversation_history: Initial conversation history
            maintainer_agent: Maintainer agent
            user_agent: User agent
            initial_docker_results: Initial Docker validation results
            exploration_log: Exploration log from initial exploration
            
        Returns:
            Tuple of (conversation_history, total_rounds, final_satisfaction_status)
        """
        max_rounds = self.config.workflow.max_conversation_rounds
        user_satisfied = False
        final_satisfaction = {
            "satisfaction_status": SatisfactionStatus.NOT_SATISFIED,
            "satisfaction_reason": "Conversation not completed"
        }
        current_docker_results = initial_docker_results
        
        for round_num in range(max_rounds):
            if user_satisfied:
                logger.info("User is satisfied. Ending conversation.")
                break
                
            logger.info(f"Starting conversation round {round_num + 1}/{max_rounds}")
            
            # User agent responds to maintainer
            try:
                # Get style data for user response (simplified for now)
                style_data = None  # Could implement style analysis here
                
                user_response_data = await user_agent.respond_to_maintainer(
                    issue_data, conversation_history, current_docker_results, style_data
                )
                
                user_response = user_response_data["response"]
                satisfaction_status = user_response_data["satisfaction_status"]
                satisfaction_reason = user_response_data["satisfaction_reason"]
                
                # Update satisfaction tracking
                user_satisfied = (satisfaction_status == SatisfactionStatus.FULLY_SATISFIED)
                final_satisfaction = {
                    "satisfaction_status": satisfaction_status,
                    "satisfaction_reason": satisfaction_reason
                }
                
                # Add to conversation
                conversation_history.append(
                    ConversationMessage(role="user", content=user_response)
                )
                
                logger.info(f"User response (round {round_num + 1}): {len(user_response)} chars")
                logger.info(f"Satisfaction status: {satisfaction_status}")
                
                if user_satisfied:
                    logger.info("User is fully satisfied. Ending conversation.")
                    break
                    
            except InputTooLongError:
                logger.warning("Input too long error in user agent. Ending conversation.")
                break
            except Exception as e:
                logger.error(f"Error getting user agent response: {e}")
                conversation_history.append(
                    ConversationMessage(role="user", content=f"Error: Failed to get proper response. {str(e)}")
                )
            
            # Early termination check
            if round_num == max_rounds - 1:
                logger.info(f"Reached maximum conversation rounds ({max_rounds}). Ending conversation.")
                break
            
            # Maintainer agent responds
            try:
                if issue_data.dockerfile:
                    # Docker-aware response
                    logger.info("Using Docker-aware maintainer response")
                    maintainer_response, extra_files, modified_dockerfile = await maintainer_agent.generate_docker_response(
                        repo_dir, issue_data, conversation_history
                    )
                    
                    # Update issue data with modifications
                    if modified_dockerfile:
                        issue_data.dockerfile = modified_dockerfile
                        logger.info("Updated Dockerfile with maintainer's modifications")
                    
                    # Run Docker validation if changes were made
                    if extra_files or modified_dockerfile:
                        logger.info("Running Docker build with maintainer's changes")
                        docker_result = await self._run_docker_validation(
                            issue_data, maintainer_response, exploration_log, extra_files
                        )
                        current_docker_results = docker_result
                        
                        # Add Docker results to conversation
                        docker_summary = f"Docker build and test {'succeeded' if docker_result.get('success') else 'failed'}.\n\nLogs:\n{docker_result.get('logs', '')[:3000]}..."
                        full_maintainer_response = maintainer_response + "\n\n" + docker_summary
                    else:
                        full_maintainer_response = maintainer_response
                        
                    conversation_history.append(
                        ConversationMessage(role="maintainer", content=full_maintainer_response)
                    )
                else:
                    # Standard response with exploration
                    maintainer_response, exploration_results = await maintainer_agent.generate_standard_response(
                        repo_dir, issue_data, conversation_history
                    )
                    
                    conversation_history.append(
                        ConversationMessage(role="maintainer", content=maintainer_response)
                    )
                    
                    # Add exploration to log (note: exploration_log is passed by reference)  
                    # We'll use a local variable to avoid parameter modification issues
                    if exploration_results:
                        logger.info(f"Conversation round {round_num + 1} exploration results logged")
                
                logger.info(f"Maintainer response (round {round_num + 1}): {len(maintainer_response)} chars")
                
            except InputTooLongError:
                logger.warning("Input too long error in maintainer agent. Ending conversation.")
                break
            except Exception as e:
                logger.error(f"Error getting maintainer agent response: {e}")
                conversation_history.append(
                    ConversationMessage(role="maintainer", content=f"Error: Failed to get proper response. {str(e)}")
                )
        
        total_rounds = round_num + 1
        logger.info(f"Conversation completed after {total_rounds} rounds")
        
        return conversation_history, total_rounds, final_satisfaction
    
    def _extract_final_maintainer_answer(self, conversation_history: List[ConversationMessage]) -> str:
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
    
    async def _run_docker_validation(
        self,
        issue_data: IssueData,
        maintainer_response: str,
        exploration_log: str,
        extra_files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run Docker validation for the current solution.
        
        Args:
            issue_data: Issue data
            maintainer_response: Maintainer's response
            exploration_log: Exploration results
            extra_files: Additional files to include
            
        Returns:
            Dictionary with Docker validation results
        """
        try:
            # Generate test commands (simplified for now)
            test_commands = []  # Would implement test command generation here
            
            # Update extra files in issue data
            if extra_files:
                if not hasattr(issue_data, 'extra_files'):
                    issue_data.extra_files = {}
                if issue_data.extra_files is None:
                    issue_data.extra_files = {}
                issue_data.extra_files.update(extra_files)
            
            # Validate Docker solution
            docker_result = self.docker_manager.validate_docker_solution(issue_data, test_commands)
            
            return {
                'success': docker_result.success,
                'logs': docker_result.logs,
                'test_commands': docker_result.test_commands,
                'error': docker_result.error
            }
            
        except Exception as e:
            logger.error(f"Error during Docker validation: {e}")
            return {
                'success': False,
                'logs': f"Docker validation error: {str(e)}",
                'test_commands': [],
                'error': str(e)
            }
