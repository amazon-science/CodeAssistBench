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
    
    def _flush_logger(self, log: logging.Logger):
        """Explicitly flush all handlers of a logger to ensure immediate writes.
        
        Args:
            log: Logger instance to flush
        """
        try:
            for handler in log.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
        except Exception as e:
            logger.error(f"Error flushing logger: {e}")
        
    async def run_generation(
        self,
        issue_data: IssueData,
        agent_model_mapping: Optional[Dict[str, str]] = None,
        agent_framework_mapping: Optional[Dict[str, str]] = None,
        issue_logger: Optional[logging.Logger] = None,
        enable_ast_tools: bool = True
    ) -> GenerationResult:
        """Run generation workflow for an issue.
        
        Args:
            issue_data: Issue data to process
            agent_model_mapping: Optional mapping of agent types to model names
            agent_framework_mapping: Optional mapping of agent types to frameworks
                                    Example: {"maintainer": "openhands"}
            issue_logger: Optional dedicated logger for this issue
            enable_ast_tools: Enable AST tools for OpenHands agent (default: True)
            
        Returns:
            GenerationResult with conversation and exploration data
        """
        # Use issue logger if provided, otherwise use default logger
        log = issue_logger or logger
        
        log.info(f"Starting generation workflow for issue: {issue_data.first_question.title}")
        self._flush_logger(log)
        
        # Create agents with framework selection
        if agent_framework_mapping:
            agents = self._create_agents_with_frameworks(
                agent_model_mapping, agent_framework_mapping, log, enable_ast_tools=enable_ast_tools
            )
        elif agent_model_mapping:
            agents = self.agent_factory.update_model_mapping(agent_model_mapping)
        else:
            agents = self.agent_factory.create_agent_set()
        
        maintainer_agent = agents["maintainer"]
        user_agent = agents["user"]
        
        # Track which framework is being used
        framework_used = agent_framework_mapping.get("maintainer", "strands") if agent_framework_mapping else "strands"
        is_openhands = framework_used == "openhands"
        log.info(f"🤖 Maintainer agent framework: {framework_used}")
        self._flush_logger(log)
        
        # Reset call counter for this issue
        maintainer_agent.reset_call_counter(issue_data.id)
        
        # Clone repository for exploration
        repo_url = self.repository_manager.parse_repo_name(issue_data.commit_info.repository)
        
        # Let maintainer choose commit
        question = f"{issue_data.first_question.title}\n\n{issue_data.first_question.body}"
        selected_commit = await maintainer_agent.choose_commit(
            issue_data.commit_info.sha, question
        )
        
        log.info(f"Selected commit for exploration: {selected_commit}")
        self._flush_logger(log)
        
        # Clone repository
        repo_dir = self.repository_manager.clone_repository(repo_url, selected_commit)
        if not repo_dir:
            raise CABEvaluationError(f"Failed to clone repository: {repo_url}")
        
        try:
            # OpenHands has its own agentic loop, skip manual iteration
            if is_openhands:
                log.info("Using OpenHands native agentic loop")
                self._flush_logger(log)
                initial_answer, exploration_history, exploration_log = await self._openhands_exploration(
                    repo_dir, question, maintainer_agent, issue_data.id, issue_logger
                )
            else:
                # Perform interactive exploration for non-OpenHands agents
                log.info("Starting interactive exploration")
                self._flush_logger(log)
                initial_answer, exploration_history, exploration_log = await self._interactive_exploration(
                    repo_dir, question, maintainer_agent, issue_data.id, issue_logger
                )
            
            log.info(f"Exploration complete. Initial answer length: {len(initial_answer)}")
            self._flush_logger(log)
            
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
                log.info("Running initial Docker validation...")
                self._flush_logger(log)
                docker_results = await self._run_docker_validation(
                    issue_data, initial_answer, exploration_log, issue_logger
                )
                self._flush_logger(log)
            
            # Conduct conversation between agents
            log.info("Starting agent conversation")
            self._flush_logger(log)
            final_conversation, total_rounds, final_satisfaction = await self._conduct_conversation(
                repo_dir, issue_data, conversation_history, maintainer_agent, user_agent, docker_results, exploration_log, issue_logger
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
            
            # Get token usage from OpenHands maintainer agent
            elif hasattr(maintainer_agent, 'get_token_usage'):
                try:
                    token_usage = maintainer_agent.get_token_usage()
                    prompt_cache_metrics["maintainer"] = {
                        "input_tokens": token_usage.get("input_tokens", 0),
                        "output_tokens": token_usage.get("output_tokens", 0),
                        "reasoning_tokens": token_usage.get("reasoning_tokens", 0),
                        "total_cost_usd": token_usage.get("total_cost_usd", 0.0),
                        "cache_hit_rate_percent": token_usage.get("cache_hit_percent", 0.0)
                    }
                except Exception as e:
                    logger.warning(f"Failed to collect OpenHands maintainer token metrics: {e}")
            
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
                prompt_cache=prompt_cache_metrics,
                agent_framework_used=self.config.agent_framework_config.maintainer_framework,
                qcli_metadata=maintainer_agent.get_qcli_metadata() if hasattr(maintainer_agent, 'get_qcli_metadata') else None
            )
            
            log.info(f"Generation workflow complete for issue {issue_data.id}")
            log.info(f"User satisfied: {result.user_satisfied}")
            log.info(f"Total conversation rounds: {total_rounds}")
            log.info(f"Total LLM calls: {sum(llm_call_stats.values())}")
            self._flush_logger(log)
            
            return result
            
        finally:
            # Cleanup repository
            self.repository_manager.cleanup_repository(repo_dir)
    
    def _create_agents_with_frameworks(
        self,
        model_mapping: Optional[Dict[str, str]],
        framework_mapping: Dict[str, str],
        logger_instance: logging.Logger,
        enable_ast_tools: bool = True
    ) -> Dict[str, Any]:
        """Create agents with framework selection.
        
        Args:
            model_mapping: Mapping of agent types to model names
            framework_mapping: Mapping of agent types to frameworks
            logger_instance: Logger instance for logging
            enable_ast_tools: Enable AST tools for OpenHands agent
            
        Returns:
            Dictionary of agent instances
        """
        agents = {}
        
        # Create maintainer with framework selection
        maintainer_framework = framework_mapping.get("maintainer", "strands")
        maintainer_model = model_mapping.get("maintainer") if model_mapping else None
        
        logger_instance.info(f"Creating maintainer agent with framework: {maintainer_framework} (AST tools: {enable_ast_tools})")
        
        agents["maintainer"] = self.agent_factory.create_maintainer_agent(
            model_name=maintainer_model,
            framework=maintainer_framework,
            openhands_config=self.config.agent_framework.openhands_config_path,
            enable_ast_tools=enable_ast_tools
        )
        
        # User and judge always use Strands (only maintainer can use OpenHands)
        user_model = model_mapping.get("user") if model_mapping else None
        agents["user"] = self.agent_factory.create_user_agent(user_model)
        
        logger_instance.info("Agent set created with framework selection")
        return agents
    
    async def _openhands_exploration(
        self,
        repo_dir: str,
        question: str,
        maintainer_agent,
        issue_id: str,
        issue_logger: Optional[logging.Logger] = None
    ) -> Tuple[str, List[str], str]:
        """Let OpenHands handle exploration with its native agentic loop.
        
        Args:
            repo_dir: Repository directory
            question: User's question
            maintainer_agent: OpenHands maintainer agent instance
            issue_id: Issue ID for tracking
            issue_logger: Optional dedicated logger for this issue
            
        Returns:
            Tuple of (final_answer, exploration_history, exploration_log)
        """
        log = issue_logger or logger
        
        system_prompt = maintainer_agent.get_system_prompt()
        user_prompt = f"Question: {question}\n\nPlease explore the repository and provide a comprehensive answer."
        
        try:
            answer = await maintainer_agent.call_llm(
                user_prompt, system_prompt, issue_id, issue_logger=log, repo_dir=repo_dir
            )
            
            if answer in ("ERROR_INPUT_TOO_LONG", "ERROR_CONTEXT_LENGTH_EXCEEDED"):
                log.warning(f"OpenHands exploration hit limit: {answer}")
                answer = "After repository exploration, I encountered context limitations. Based on the exploration conducted, I can provide relevant information about this issue."
            
            return answer, ["OpenHands native exploration"], "OpenHands handled exploration internally"
            
        except Exception as e:
            log.error(f"OpenHands exploration error: {e}")
            return f"Error during exploration: {e}", [], str(e)
    
    async def _interactive_exploration(
        self,
        repo_dir: str,
        question: str,
        maintainer_agent,
        issue_id: str,
        issue_logger: Optional[logging.Logger] = None,
        max_iterations: int = 5
    ) -> Tuple[str, List[str], str]:
        """Perform interactive repository exploration.
        
        Args:
            repo_dir: Repository directory
            question: User's question
            maintainer_agent: Maintainer agent instance
            issue_id: Issue ID for tracking
            issue_logger: Optional dedicated logger for this issue
            max_iterations: Maximum exploration iterations
            
        Returns:
            Tuple of (final_answer, exploration_history, exploration_log)
        """
        # Use issue logger if provided, otherwise use default logger
        log = issue_logger or logger
        
        exploration_history = []
        exploration_log = ""
        
        # Estimate context size limits (150K tokens ~= 600K chars)
        max_context_size = 600000
        current_exploration_context = ""
        
        log.info(f"Starting interactive exploration with max {max_iterations} iterations")
        self._flush_logger(log)
        
        for iteration in range(max_iterations):
            log.info(f"Exploration iteration {iteration+1}/{max_iterations}")
            
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
                    user_prompt, system_prompt, issue_id, issue_logger=log, repo_dir=repo_dir
                )
                
                # Check for input too long error
                if exploration_plan == "ERROR_INPUT_TOO_LONG":
                    log.warning("Input too long error. Stopping exploration.")
                    exploration_log += "\n--- EXPLORATION STOPPED: Input too long error ---\n"
                    break
                
                # Check for context length exceeded error
                if exploration_plan == "ERROR_CONTEXT_LENGTH_EXCEEDED":
                    log.warning("⚠️ Context length exceeded. Stopping exploration to proceed to judge.")
                    exploration_log += "\n--- EXPLORATION STOPPED: Context length exceeded ---\n"
                    break
                
                exploration_history.append(exploration_plan)
                log.info(f"Received exploration plan ({len(exploration_plan)} chars)")
                self._flush_logger(log)
                
            except InputTooLongError:
                log.warning("Input too long error in exploration. Stopping exploration.")
                exploration_log += "\n--- EXPLORATION STOPPED: Input too long error ---\n"
                break
            except Exception as e:
                error_str = str(e).lower()
                # Check for context length error in exception
                if "max_tokens" in error_str or "max_completion_tokens" in error_str or "context length" in error_str:
                    log.warning("⚠️ Context length exceeded (exception). Stopping exploration to proceed to judge.")
                    exploration_log += "\n--- EXPLORATION STOPPED: Context length exceeded ---\n"
                    break
                log.error(f"Error getting exploration plan: {e}")
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
                
                log.info(f"Executing {len(commands)} exploration commands")
                self._flush_logger(log)
                
                for i, cmd in enumerate(commands):
                    try:
                        log.info(f"Executing command {i+1}/{len(commands)}: {cmd}")
                        result = execute_command(repo_dir, cmd, timeout=self.config.workflow.command_timeout)
                        iteration_results += f"Command: {cmd}\nResult:\n{result}\n\n"
                    except Exception as e:
                        error_msg = f"Error executing command: {cmd}\nError: {str(e)}\n\n"
                        log.error(f"Command execution error: {str(e)}")
                        iteration_results += error_msg
            
            # Add iteration results to full log
            exploration_log += f"\n--- ITERATION {iteration+1} ---\n{iteration_results}"
            
            # Check context size before adding to current context
            new_context = current_exploration_context + f"\n--- ITERATION {iteration+1} ---\n{iteration_results}"
            
            if len(new_context) > max_context_size:
                log.warning(f"Exploration context would exceed size limit. Stopping exploration.")
                exploration_log += "\n--- EXPLORATION STOPPED: Context size limit ---\n"
                break
            else:
                current_exploration_context = new_context
            
            # Check for answer
            if "ANSWER:" in exploration_plan:
                log.info("Found ANSWER section. Extracting final answer.")
                self._flush_logger(log)
                answer_part = exploration_plan.split("ANSWER:", 1)[1].strip()
                return answer_part, exploration_history, exploration_log
        
        # Generate final answer if no explicit answer found
        log.info("Generating final answer from exploration results")
        self._flush_logger(log)
        final_system_prompt = maintainer_agent.get_system_prompt() + TaskPrompts.FINAL_ANSWER_GENERATION
        final_user_prompt = f"""
        Question: {question}
        
        Exploration results:
        {current_exploration_context}
        
        Please provide a comprehensive answer based on the exploration results above.
        """
        
        try:
            final_answer = await maintainer_agent.call_llm(
                final_user_prompt, final_system_prompt, issue_id, issue_logger=log, repo_dir=repo_dir
            )
            
            if final_answer == "ERROR_INPUT_TOO_LONG":
                log.warning("Input too long in final answer generation. Using fallback.")
                final_answer = "After extensive repository exploration, I encountered context limitations. Based on the exploration conducted, I can provide relevant information about this issue."
            
            if final_answer == "ERROR_CONTEXT_LENGTH_EXCEEDED":
                log.warning("⚠️ Context length exceeded in final answer generation. Using fallback.")
                final_answer = "After repository exploration, I encountered context length limitations. Based on the exploration conducted, I can provide relevant information about this issue."
            
            return final_answer, exploration_history, exploration_log
            
        except InputTooLongError:
            log.warning("Input too long in final answer generation. Using fallback.")
            final_answer = "After extensive repository exploration, I encountered context limitations. Based on the exploration conducted, I can provide relevant information about this issue."
            return final_answer, exploration_history, exploration_log
        except Exception as e:
            error_str = str(e).lower()
            # Check for context length error in exception
            if "max_tokens" in error_str or "max_completion_tokens" in error_str or "context length" in error_str:
                log.warning("⚠️ Context length exceeded (exception) in final answer. Using fallback.")
                final_answer = "After repository exploration, I encountered context length limitations. Based on the exploration conducted, I can provide relevant information about this issue."
                return final_answer, exploration_history, exploration_log
            log.error(f"Error generating final answer: {e}")
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
        exploration_log: str = "",
        issue_logger: Optional[logging.Logger] = None
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
            issue_logger: Optional dedicated logger for this issue
            
        Returns:
            Tuple of (conversation_history, total_rounds, final_satisfaction_status)
        """
        # Use issue logger if provided, otherwise use default logger
        log = issue_logger or logger
        
        max_rounds = self.config.workflow.max_conversation_rounds
        user_satisfied = False
        final_satisfaction = {
            "satisfaction_status": SatisfactionStatus.NOT_SATISFIED,
            "satisfaction_reason": "Conversation not completed"
        }
        current_docker_results = initial_docker_results
        
        for round_num in range(max_rounds):
            if user_satisfied:
                log.info("User is satisfied. Ending conversation.")
                break
                
            log.info(f"Starting conversation round {round_num + 1}/{max_rounds}")
            
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
                
                log.info(f"User response (round {round_num + 1}): {len(user_response)} chars")
                log.info(f"Satisfaction status: {satisfaction_status}")
                self._flush_logger(log)
                
                if user_satisfied:
                    log.info("User is fully satisfied. Ending conversation.")
                    break
                    
            except InputTooLongError:
                log.warning("Input too long error in user agent. Ending conversation.")
                break
            except Exception as e:
                log.error(f"Error getting user agent response: {e}")
                conversation_history.append(
                    ConversationMessage(role="user", content=f"Error: Failed to get proper response. {str(e)}")
                )
            
            # Early termination check
            if round_num == max_rounds - 1:
                log.info(f"Reached maximum conversation rounds ({max_rounds}). Ending conversation.")
                break
            
            # Maintainer agent responds
            try:
                if issue_data.dockerfile:
                    # Docker-aware response
                    log.info("Using Docker-aware maintainer response")
                    maintainer_response, extra_files, modified_dockerfile = await maintainer_agent.generate_docker_response(
                        repo_dir, issue_data, conversation_history, issue_logger=log
                    )
                    
                    # Check for context length exceeded error
                    if maintainer_response == "ERROR_CONTEXT_LENGTH_EXCEEDED":
                        log.warning("⚠️ Context length exceeded in maintainer response. Ending conversation to proceed to judge.")
                        self._flush_logger(log)
                        break
                    
                    # Update issue data with modifications
                    if modified_dockerfile:
                        issue_data.dockerfile = modified_dockerfile
                        log.info("Updated Dockerfile with maintainer's modifications")
                    
                    # Run Docker validation if changes were made
                    if extra_files or modified_dockerfile:
                        log.info("Running Docker build with maintainer's changes")
                        docker_result = await self._run_docker_validation(
                            issue_data, maintainer_response, exploration_log, extra_files, issue_logger
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
                        repo_dir, issue_data, conversation_history, issue_logger=log
                    )
                    
                    # Check for context length exceeded error
                    if maintainer_response == "ERROR_CONTEXT_LENGTH_EXCEEDED":
                        log.warning("⚠️ Context length exceeded in maintainer response. Ending conversation to proceed to judge.")
                        self._flush_logger(log)
                        break
                    
                    conversation_history.append(
                        ConversationMessage(role="maintainer", content=maintainer_response)
                    )
                    
                    # Add exploration to log (note: exploration_log is passed by reference)  
                    # We'll use a local variable to avoid parameter modification issues
                    if exploration_results:
                        log.info(f"Conversation round {round_num + 1} exploration results logged")
                
                log.info(f"Maintainer response (round {round_num + 1}): {len(maintainer_response)} chars")
                self._flush_logger(log)
                
            except InputTooLongError:
                log.warning("Input too long error in maintainer agent. Ending conversation.")
                break
            except Exception as e:
                error_str = str(e).lower()
                # Check for context length error in exception
                if "max_tokens" in error_str or "max_completion_tokens" in error_str or "context length" in error_str:
                    log.warning("⚠️ Context length exceeded (exception). Ending conversation to proceed to judge.")
                    self._flush_logger(log)
                    break
                log.error(f"Error getting maintainer agent response: {e}")
                conversation_history.append(
                    ConversationMessage(role="maintainer", content=f"Error: Failed to get proper response. {str(e)}")
                )
        
        total_rounds = round_num + 1
        log.info(f"Conversation completed after {total_rounds} rounds")
        self._flush_logger(log)
        
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
        extra_files: Optional[Dict[str, str]] = None,
        issue_logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """Run Docker validation for the current solution.
        
        Args:
            issue_data: Issue data
            maintainer_response: Maintainer's response
            exploration_log: Exploration results
            extra_files: Additional files to include
            issue_logger: Optional dedicated logger for this issue
            
        Returns:
            Dictionary with Docker validation results
        """
        # Use issue logger if provided, otherwise use default logger
        log = issue_logger or logger
        
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
            log.error(f"Error during Docker validation: {e}")
            return {
                'success': False,
                'logs': f"Docker validation error: {str(e)}",
                'test_commands': [],
                'error': str(e)
            }
