"""Command-line interface for CAB evaluation."""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .core.config import CABConfig
from .utils.data_processor import DataProcessor
from .workflows.cab_workflow import CABWorkflow


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers
    )


async def run_dataset(args):
    """Run CAB evaluation on a dataset."""
    logger = logging.getLogger(__name__)
    
    # Load configuration  
    config = None
    if args.config:
        config = CABConfig.from_file(args.config)
    
    # Parse agent model mapping
    agent_model_mapping = None
    if args.agent_models:
        try:
            agent_model_mapping = json.loads(args.agent_models)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in agent_models: {e}")
            return 1
    
    # Run dataset processing
    try:
        evaluator = CABWorkflow(config)
        summary = await evaluator.process_dataset(
            dataset_path=args.dataset_path,
            target_language=args.language,
            output_dir=args.output_dir,
            agent_model_mapping=agent_model_mapping,
            batch_size=args.batch_size,
            resume_processing=args.resume
        )
        
        logger.info(f"Dataset processing complete: {summary}")
        return 0
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        return 1


async def run_generation_dataset(args):
    """Run generation workflow on JSONL dataset."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = None
    if args.config:
        config = CABConfig.from_file(args.config)
    
    # Parse agent model mapping
    agent_model_mapping = None
    if args.agent_models:
        try:
            agent_model_mapping = json.loads(args.agent_models)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in agent_models: {e}")
            return 1
    
    # Create output filename if not specified
    if not args.output:
        dataset_name = Path(args.dataset_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create folder structure: results/generation/
        output_dir = Path("results/generation")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"generation_results_{dataset_name}_{timestamp}.jsonl")
    
    # Load all issues from JSONL
    try:
        data_processor = DataProcessor()
        raw_data = data_processor.load_jsonl_data([args.dataset_file])
        
        # Filter by language if specified
        if args.language:
            raw_data = [item for item in raw_data if item.get('language', '').lower() == args.language.lower()]
            logger.info(f"Filtered to {len(raw_data)} issues for language: {args.language}")
        
        # Convert to IssueData objects
        issues = []
        for i, item in enumerate(raw_data):
            try:
                issue_data = data_processor.load_issue_data_from_dict(item)
                issues.append(issue_data)
            except Exception as e:
                logger.warning(f"Failed to load issue {i}: {e}")
                continue
        
        logger.info(f"Loaded {len(issues)} valid issues from {args.dataset_file}")
        
    except Exception as e:
        logger.error(f"Error loading JSONL dataset: {e}")
        return 1
    
    # Check for resume functionality
    processed_issues = set()
    if args.resume and Path(args.output).exists():
        try:
            with open(args.output, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if 'issue_id' in result:
                            processed_issues.add(result['issue_id'])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Resuming: found {len(processed_issues)} already processed issues")
        except Exception as e:
            logger.warning(f"Error reading existing output file for resume: {e}")
    
    # Filter out already processed issues
    issues_to_process = [issue for issue in issues if issue.id not in processed_issues]
    logger.info(f"Processing {len(issues_to_process)} issues (total: {len(issues)})")
    
    # Process issues and write results to JSONL
    try:
        from .workflows.generation_workflow import GenerationWorkflow
        
        generation_workflow = GenerationWorkflow(config)
        successful_count = 0
        failed_count = 0
        
        # Open output file for appending (for resume functionality)
        mode = 'a' if (args.resume and Path(args.output).exists()) else 'w'
        with open(args.output, mode) as f:
            for i, issue_data in enumerate(issues_to_process):
                logger.info(f"Processing issue {i+1}/{len(issues_to_process)}: {issue_data.id} - {issue_data.first_question.title}")
                
                try:
                    # Run generation workflow
                    result = await generation_workflow.run_generation(issue_data, agent_model_mapping)
                    
                    # Get original issue data for complete metadata preservation
                    original_issue = None
                    for orig_item in raw_data:
                        if str(orig_item.get('number')) == str(result.issue_data.id):
                            original_issue = orig_item
                            break
                    
                    # Convert to dictionary with complete metadata including all original fields
                    result_dict = {
                        # Core result data
                        'issue_id': result.issue_data.id,
                        'question_title': result.issue_data.first_question.title,
                        'question_body': result.issue_data.first_question.body,
                        'user': result.issue_data.first_question.user,
                        'language': result.issue_data.language,
                        'repository': result.issue_data.commit_info.repository,
                        'commit_sha': result.issue_data.commit_info.sha,
                        'final_answer': result.final_answer,
                        'user_satisfied': result.user_satisfied,
                        'satisfaction_status': result.satisfaction_status.value,
                        'satisfaction_reason': result.satisfaction_reason,
                        'total_conversation_rounds': result.total_conversation_rounds,
                        'original_comment_count': result.original_comment_count,
                        'conversation_history': [
                            {
                                'role': msg.role,
                                'content': msg.content
                            }
                            for msg in result.conversation_history
                        ],
                        'exploration_history': result.exploration_history,
                        'exploration_log': result.exploration_log,
                        'llm_call_counter': result.llm_call_counter,
                        'prompt_cache': result.prompt_cache,
                        
                        # Original input metadata preservation - ALL fields from input JSONL
                        'original_metadata': original_issue if original_issue else {},
                        
                        'processing_metadata': {
                            'workflow': 'generation_only',
                            'timestamp': datetime.now().isoformat(),
                            'agent_model_mapping': agent_model_mapping or {},
                            'input_file': args.dataset_file,
                            'language_filter': args.language
                        }
                    }
                    
                    # Write result as JSONL line
                    f.write(json.dumps(result_dict, default=str) + '\n')
                    f.flush()  # Ensure immediate write for progress tracking
                    
                    successful_count += 1
                    logger.info(f"✅ Issue {issue_data.id} processed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing issue {issue_data.id}: {e}")
                    
                    # Get original issue data for complete metadata preservation in error case
                    original_issue = None
                    for orig_item in raw_data:
                        if str(orig_item.get('number')) == str(issue_data.id):
                            original_issue = orig_item
                            break
                    
                    # Write error result to maintain JSONL consistency with complete metadata
                    error_result = {
                        'issue_id': issue_data.id,
                        'question_title': issue_data.first_question.title,
                        'question_body': issue_data.first_question.body,
                        'user': issue_data.first_question.user,
                        'language': issue_data.language,
                        'repository': issue_data.commit_info.repository,
                        'commit_sha': issue_data.commit_info.sha,
                        'error': str(e),
                        'user_satisfied': False,
                        'satisfaction_status': 'ERROR',
                        'satisfaction_reason': f'Processing failed: {str(e)}',
                        'total_conversation_rounds': 0,
                        
                        # Original input metadata preservation for error cases too - ALL fields
                        'original_metadata': original_issue if original_issue else {},
                        
                        'processing_metadata': {
                            'workflow': 'generation_only',
                            'timestamp': datetime.now().isoformat(),
                            'error_occurred': True,
                            'error_message': str(e),
                            'input_file': args.dataset_file
                        }
                    }
                    f.write(json.dumps(error_result, default=str) + '\n')
                    f.flush()
                    
                    failed_count += 1
        
        # Log final summary
        logger.info(f"=== GENERATION DATASET PROCESSING COMPLETE ===")
        logger.info(f"Total issues processed: {len(issues_to_process)}")
        logger.info(f"Successful: {successful_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation dataset processing failed: {e}")
        return 1


async def run_evaluation_dataset(args):
    """Run evaluation workflow on JSONL generation results."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = None
    if args.config:
        config = CABConfig.from_file(args.config)
    
    # Parse agent model mapping
    agent_model_mapping = None
    if args.agent_models:
        try:
            agent_model_mapping = json.loads(args.agent_models)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in agent_models: {e}")
            return 1
    
    # Create output filename if not specified
    if not args.output:
        input_name = Path(args.generation_results_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create folder structure: results/evaluation/
        output_dir = Path("results/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"evaluation_results_{input_name}_{timestamp}.jsonl")
    
    # Load all generation results from JSONL
    try:
        generation_results = []
        with open(args.generation_results_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    generation_dict = json.loads(line.strip())
                    generation_results.append(generation_dict)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(generation_results)} generation results from {args.generation_results_file}")
        
    except Exception as e:
        logger.error(f"Error loading generation results JSONL: {e}")
        return 1
    
    # Check for resume functionality
    processed_issues = set()
    if args.resume and Path(args.output).exists():
        try:
            with open(args.output, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if 'issue_id' in result:
                            processed_issues.add(result['issue_id'])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Resuming: found {len(processed_issues)} already processed evaluations")
        except Exception as e:
            logger.warning(f"Error reading existing output file for resume: {e}")
    
    # Filter out already processed results
    results_to_process = [result for result in generation_results if result.get('issue_id') not in processed_issues]
    logger.info(f"Processing {len(results_to_process)} evaluations (total: {len(generation_results)})")
    
    # Process generation results and write evaluation results to JSONL
    try:
        from .workflows.evaluation_workflow import EvaluationWorkflow
        from .core.models import GenerationResult, ConversationMessage, SatisfactionStatus, JudgeConfig
        
        # Create judge config for iterative evaluation if enabled
        judge_config = JudgeConfig() if getattr(args, 'iterative', False) else None
        evaluation_workflow = EvaluationWorkflow(config, judge_config=judge_config)
        
        # Log evaluation mode
        if judge_config:
            logger.info(f"🔄 Using iterative judge evaluation (max_iterations={judge_config.max_iterations}, repo_exploration={judge_config.enable_repository_exploration})")
        else:
            logger.info("📍 Using traditional single-iteration judge evaluation")
        data_processor = DataProcessor()
        successful_count = 0
        failed_count = 0
        
        # Open output file for appending (for resume functionality)
        mode = 'a' if (args.resume and Path(args.output).exists()) else 'w'
        with open(args.output, mode) as f:
            for i, generation_dict in enumerate(results_to_process):
                issue_id = generation_dict.get('issue_id', f'unknown_{i}')
                logger.info(f"Evaluating {i+1}/{len(results_to_process)}: {issue_id} - {generation_dict.get('question_title', 'Unknown')}")
                
                try:
                    # Use original_metadata directly for reconstruction (simpler and more reliable)
                    original_metadata = generation_dict.get('original_metadata', {})
                    
                    # Build issue_data_dict using original_metadata which has all the correct info
                    issue_data_dict = original_metadata.copy()  # Start with original data
                    
                    # Override with generation result data where needed
                    issue_data_dict.update({
                        'id': generation_dict['issue_id'],
                        'number': original_metadata.get('number', generation_dict['issue_id']),  # Ensure number field exists
                        'language': generation_dict.get('language', original_metadata.get('language', 'unknown')),
                        'title': generation_dict.get('question_title', original_metadata.get('title', '')),
                        'body': generation_dict.get('question_body', original_metadata.get('body', '')),
                        'author': generation_dict.get('user', original_metadata.get('author', 'unknown')),
                        'satisfaction_conditions': original_metadata.get('satisfaction_conditions', []),
                    })
                    
                    # Ensure we have the required fields for data processor
                    if 'url' not in issue_data_dict and 'repository' in generation_dict:
                        issue_data_dict['url'] = generation_dict['repository']
                    if 'commit_id' not in issue_data_dict and 'commit_sha' in generation_dict:
                        issue_data_dict['commit_id'] = generation_dict['commit_sha']
                    
                    issue_data = data_processor.load_issue_data_from_dict(issue_data_dict)
                    
                    # Reconstruct GenerationResult
                    conversation_history = []
                    for msg_dict in generation_dict.get('conversation_history', []):
                        conversation_history.append(
                            ConversationMessage(role=msg_dict['role'], content=msg_dict['content'])
                        )
                    
                    satisfaction_status = SatisfactionStatus(generation_dict.get('satisfaction_status', 'NOT_SATISFIED'))
                    
                    generation_result = GenerationResult(
                        issue_data=issue_data,
                        modified_dockerfile=generation_dict.get('modified_dockerfile'),
                        total_conversation_rounds=generation_dict.get('total_conversation_rounds', 0),
                        original_comment_count=generation_dict.get('original_comment_count', 0),
                        user_satisfied=generation_dict.get('user_satisfied', False),
                        exploration_history=generation_dict.get('exploration_history', []),
                        exploration_log=generation_dict.get('exploration_log', ''),
                        conversation_history=conversation_history,
                        llm_call_counter=generation_dict.get('llm_call_counter', {}),
                        satisfaction_status=satisfaction_status,
                        satisfaction_reason=generation_dict.get('satisfaction_reason', ''),
                        final_answer=generation_dict.get('final_answer', ''),
                        prompt_cache=generation_dict.get('prompt_cache', {})
                    )
                    
                    # Run evaluation workflow (iterative if judge_config is provided)
                    if judge_config:
                        # Use iterative evaluation with repository path
                        repository_path = generation_dict.get('original_metadata', {}).get('repository_path', str(Path.cwd()))
                        evaluation_result = await evaluation_workflow.run_iterative_evaluation(
                            generation_result, 
                            repository_path, 
                            agent_model_mapping
                        )
                    else:
                        # Use traditional single-iteration evaluation
                        evaluation_result = await evaluation_workflow.run_evaluation(generation_result, agent_model_mapping)
                    
                    # Convert to dictionary with complete metadata
                    result_dict = {
                        # Core evaluation result data
                        'issue_id': issue_data.id,
                        'question_title': issue_data.first_question.title,
                        'question_body': issue_data.first_question.body,
                        'user': issue_data.first_question.user,
                        'language': issue_data.language,
                        'repository': issue_data.commit_info.repository,
                        'commit_sha': issue_data.commit_info.sha,
                        'final_answer': generation_result.final_answer,
                        'judgment': evaluation_result.judgment,
                        'verdict': evaluation_result.verdict.value,
                        'key_issues': evaluation_result.key_issues,
                        'is_iterative': evaluation_result.is_iterative,
                        'alignment_score': (
                            {
                                'satisfied': evaluation_result.alignment_score.satisfied,
                                'total': evaluation_result.alignment_score.total,
                                'percentage': evaluation_result.alignment_score.percentage,
                                'conditions': [
                                    {
                                        'number': cond.number,
                                        'satisfied': cond.satisfied,
                                        'description': cond.description
                                    }
                                    for cond in evaluation_result.alignment_score.conditions
                                ]
                            } if evaluation_result.alignment_score else None
                        ),
                        'docker_results': (
                            {
                                'success': evaluation_result.docker_results.success,
                                'logs': evaluation_result.docker_results.logs,
                                'test_commands': evaluation_result.docker_results.test_commands,
                                'error': evaluation_result.docker_results.error
                            } if evaluation_result.docker_results else None
                        ),
                        'llm_calls': evaluation_result.llm_calls,
                        'prompt_cache': evaluation_result.prompt_cache,
                        
                        # Iterative evaluation metadata (if available)
                        'iterative_evaluation': (
                            {
                                'iterations_completed': len(evaluation_result.iterative_evaluation.iterations),
                                'max_iterations': judge_config.max_iterations if judge_config else 1,
                                'total_evaluation_time_seconds': evaluation_result.iterative_evaluation.total_evaluation_time_seconds,
                                'stopped_early': evaluation_result.iterative_evaluation.stopped_early,
                                'early_stopping_reason': evaluation_result.iterative_evaluation.early_stopping_reason,
                                'confidence_progression': evaluation_result.iterative_evaluation.confidence_progression,
                                'repository_exploration': (
                                    {
                                        'files_read': evaluation_result.iterative_evaluation.repository_exploration.files_read,
                                        'exploration_time_seconds': evaluation_result.iterative_evaluation.repository_exploration.exploration_time_seconds,
                                        'total_files_found': evaluation_result.iterative_evaluation.repository_exploration.total_files_found
                                    } if evaluation_result.iterative_evaluation.repository_exploration else None
                                ),
                                'conversation_analysis': evaluation_result.iterative_evaluation.conversation_analysis,
                                'total_token_usage': evaluation_result.iterative_evaluation.total_token_usage
                            } if evaluation_result.iterative_evaluation else None
                        ),
                        
                        # Generation result metadata
                        'generation_metadata': {
                            'user_satisfied': generation_result.user_satisfied,
                            'satisfaction_status': generation_result.satisfaction_status.value,
                            'satisfaction_reason': generation_result.satisfaction_reason,
                            'total_conversation_rounds': generation_result.total_conversation_rounds,
                            'generation_llm_calls': generation_result.llm_call_counter
                        },
                        
                        # Original input metadata preservation
                        'original_metadata': generation_dict.get('original_metadata', {}),
                        
                        'processing_metadata': {
                            'workflow': 'evaluation_iterative' if judge_config else 'evaluation_only',
                            'timestamp': datetime.now().isoformat(),
                            'agent_model_mapping': agent_model_mapping or {},
                            'input_file': args.generation_results_file,
                            'judge_config_used': (
                                {
                                    'max_iterations': judge_config.max_iterations,
                                    'enable_repository_exploration': judge_config.enable_repository_exploration,
                                    'enable_conversation_analysis': judge_config.enable_conversation_analysis,
                                    'confidence_threshold': judge_config.confidence_threshold,
                                    'early_stopping_enabled': judge_config.early_stopping_enabled
                                } if judge_config else None
                            )
                        }
                    }
                    
                    # Write result as JSONL line
                    f.write(json.dumps(result_dict, default=str) + '\n')
                    f.flush()  # Ensure immediate write for progress tracking
                    
                    successful_count += 1
                    logger.info(f"✅ Issue {issue_id} evaluated successfully - Verdict: {evaluation_result.verdict.value}")
                    
                except Exception as e:
                    logger.error(f"❌ Error evaluating issue {issue_id}: {e}")
                    
                    # Write error result to maintain JSONL consistency
                    error_result = {
                        'issue_id': issue_id,
                        'question_title': generation_dict.get('question_title', ''),
                        'question_body': generation_dict.get('question_body', ''),
                        'user': generation_dict.get('user', ''),
                        'language': generation_dict.get('language', ''),
                        'repository': generation_dict.get('repository', ''),
                        'commit_sha': generation_dict.get('commit_sha', ''),
                        'error': str(e),
                        'verdict': 'ERROR',
                        'judgment': f'Evaluation failed: {str(e)}',
                        'key_issues': [f'Processing error: {str(e)}'],
                        
                        # Preserve metadata even in error cases
                        'original_metadata': generation_dict.get('original_metadata', {}),
                        'generation_metadata': {
                            'user_satisfied': generation_dict.get('user_satisfied', False),
                            'satisfaction_status': generation_dict.get('satisfaction_status', 'UNKNOWN'),
                            'total_conversation_rounds': generation_dict.get('total_conversation_rounds', 0)
                        },
                        
                        'processing_metadata': {
                            'workflow': 'evaluation_only',
                            'timestamp': datetime.now().isoformat(),
                            'error_occurred': True,
                            'error_message': str(e),
                            'input_file': args.generation_results_file
                        }
                    }
                    f.write(json.dumps(error_result, default=str) + '\n')
                    f.flush()
                    
                    failed_count += 1
        
        # Log final summary
        logger.info(f"=== EVALUATION DATASET PROCESSING COMPLETE ===")
        logger.info(f"Total evaluations processed: {len(results_to_process)}")
        logger.info(f"Successful: {successful_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation dataset processing failed: {e}")
        return 1


def create_config_template(args):
    """Create a configuration template file."""
    config = CABConfig()
    
    try:
        config.save_to_file(args.output)
        print(f"Configuration template created at: {args.output}")
        return 0
    except Exception as e:
        print(f"Error creating config template: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CAB Evaluation - Code Agent Benchmark evaluation package"
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    parser.add_argument(
        "--config",
        help="Configuration file path"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dataset command - Complete CAB evaluation on JSONL dataset
    dataset_parser = subparsers.add_parser("dataset", help="Run complete CAB evaluation on JSONL dataset")
    dataset_parser.add_argument(
        "dataset_path",
        help="Path to JSONL dataset file"
    )
    dataset_parser.add_argument(
        "--language", "-l",
        help="Filter by programming language"
    )
    dataset_parser.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Output directory for results"
    )
    dataset_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    dataset_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous processing (skip already processed issues)"
    )
    dataset_parser.add_argument(
        "--agent-models",
        help='JSON mapping of agents to models'
    )
    
    # Generation dataset command for JSONL files
    generation_dataset_parser = subparsers.add_parser("generation-dataset", help="Run generation workflow on JSONL dataset")
    generation_dataset_parser.add_argument(
        "dataset_file",
        help="JSONL file containing multiple issues"
    )
    generation_dataset_parser.add_argument(
        "--output", "-o",
        help="Output JSONL file for results (default: generation_results_<timestamp>.jsonl)"
    )
    generation_dataset_parser.add_argument(
        "--language", "-l",
        help="Filter by programming language"
    )
    generation_dataset_parser.add_argument(
        "--agent-models",
        help='JSON mapping of agents to models (e.g., \'{"maintainer": "sonnet37", "user": "haiku"}\')'
    )
    generation_dataset_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing (skip already processed issues)"
    )
    
    # Evaluation dataset command for JSONL files with generation results
    evaluation_dataset_parser = subparsers.add_parser("evaluation-dataset", help="Run evaluation workflow on JSONL generation results")
    evaluation_dataset_parser.add_argument(
        "generation_results_file",
        help="JSONL file containing multiple generation results"
    )
    evaluation_dataset_parser.add_argument(
        "--output", "-o",
        help="Output JSONL file for evaluation results (default: evaluation_results_<timestamp>.jsonl)"
    )
    evaluation_dataset_parser.add_argument(
        "--agent-models",
        help='JSON mapping for judge agent model (e.g., \'{"judge": "sonnet"}\')'
    )
    evaluation_dataset_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing (skip already processed issues)"
    )
    evaluation_dataset_parser.add_argument(
        "--iterative",
        action="store_true",
        help="Enable iterative judge evaluation with repository exploration and conversation analysis"
    )
    
    # Config template command
    config_parser = subparsers.add_parser("config", help="Create configuration template")
    config_parser.add_argument(
        "output",
        help="Output path for config template"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Run appropriate command
    if args.command == "dataset":
        return asyncio.run(run_dataset(args))
    elif args.command == "generation-dataset":
        return asyncio.run(run_generation_dataset(args))
    elif args.command == "evaluation-dataset":
        return asyncio.run(run_evaluation_dataset(args))
    elif args.command == "config":
        return create_config_template(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
