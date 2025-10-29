"""Example script for running CAB evaluation."""

import asyncio
import logging
from pathlib import Path

from cab_evaluation import (
    create_cab_evaluator,
    CABConfig,
    DataProcessor,
    CABEvaluationError
)


async def main():
    """Main example function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('cab_evaluation_example.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting CAB evaluation example")
    
    try:
        # Example 1: Process a single issue with default configuration
        logger.info("=== Example 1: Single Issue Processing ===")
        
        # Create evaluator with default config
        evaluator = create_cab_evaluator()
        
        # Load a sample issue (you would replace this with actual data loading)
        data_processor = DataProcessor()
        
        # For this example, create a mock issue
        sample_issue_dict = {
            "number": 67,
            "title": "[Feature require] Allow another port, not just only 3000 port",
            "created_at": "2025-01-11T07:09:51Z",
            "closed_at": "2025-01-11T13:06:09Z",
            "commit_id": "de618b55495e3ba16431079e18f7aa1a2a608b7c",
            "labels": [],
            "url": "https://github.com/elizaOS/eliza-starter/issues/67",
            "body": "I want to run multiple agent with one server. but when start single agent which occupy 3000 port, so other agent can not be launched.\r\n\r\nI checked this problem, this port occupation occurs on @ai16z/client-direct module.\r\n\r\nInside  @ai16z/client-direct module, 3000 port is hard coded. \r\n\r\n",
            "comments_url": "https://api.github.com/repos/elizaOS/eliza-starter/issues/67/comments",
            "author": "joshephan",
            "comments": [{
                "user": "divyangchauhan",
                "created_at": "2025-01-11T12:43:27Z",
                "body": "use can set SERVER_PORT in .env file to your desired port number to change the port."
            }, {
                "user": "joshephan",
                "created_at": "2025-01-11T13:06:09Z",
                "body": "@divyangchauhan Oh my mistake. it works. Thanks."
            }],
            "satisfaction_conditions": ["A way to configure the port number for running multiple agents simultaneously", "Information about existing configuration options that aren't immediately obvious in the codebase", "A solution that doesn't require code modification"],
            "_classification": {
                "category": "Does not need build environment",
                "timestamp": "2025-04-14 01:00:24"
            },
            "dockerfile": null,
            "language": "typescript"
        }
        
        # Convert to IssueData
        issue_data = data_processor.load_issue_data_from_dict(sample_issue_dict)
        
        # Run complete evaluation
        result = await evaluator.run_complete_evaluation(issue_data)
        
        # Print results
        logger.info(f"Issue: {result.question_title}")
        logger.info(f"User satisfied: {result.generation_result.user_satisfied}")
        logger.info(f"Final verdict: {result.evaluation_result.verdict}")
        logger.info(f"Conversation rounds: {result.generation_result.total_conversation_rounds}")
        
        if result.evaluation_result.alignment_score:
            score = result.evaluation_result.alignment_score
            logger.info(f"Alignment: {score.satisfied}/{score.total} conditions met ({score.percentage:.1f}%)")
        
        
        # Example 2: Process with custom agent models
        logger.info("=== Example 2: Custom Agent Models ===")
        
        # Create evaluator with custom model mapping
        custom_evaluator = create_cab_evaluator(
            agent_model_mapping={
                "maintainer": "sonnet37",  # Use Sonnet 3.7 for maintainer
                "user": "haiku",          # Use Haiku for user (faster)
                "judge": "sonnet"         # Use Sonnet for judge (accurate)
            }
        )
        
        logger.info("Created evaluator with custom model mapping")
        
        
        # Example 3: Process dataset by language
        logger.info("=== Example 3: Dataset Processing ===")
        
        # This example shows how you would process a real dataset
        dataset_path = "path/to/your/dataset.jsonl"  # Replace with actual path
        
        if Path(dataset_path).exists():
            # Process Python issues from dataset
            summary = await evaluator.process_dataset(
                dataset_path=dataset_path,
                target_language="python",
                output_dir="results/python_evaluation",
                agent_model_mapping={
                    "maintainer": "haiku",    # Fast model for maintainer
                    "user": "sonnet",        # Good model for user simulation
                    "judge": "sonnet"        # Accurate model for judgment
                },
                batch_size=5,
                resume_processing=True
            )
            
            logger.info(f"Dataset processing summary: {summary}")
        else:
            logger.info(f"Dataset file {dataset_path} not found, skipping dataset example")
        
        
        # Example 4: Using different workflows separately
        logger.info("=== Example 4: Separate Workflow Usage ===")
        
        # Use only generation workflow
        from cab_evaluation.workflows import GenerationWorkflow
        gen_workflow = GenerationWorkflow()
        
        gen_result = await gen_workflow.run_generation(issue_data)
        logger.info(f"Generation only - User satisfied: {gen_result.user_satisfied}")
        
        # Then use evaluation workflow on the generation result
        from cab_evaluation.workflows import EvaluationWorkflow
        eval_workflow = EvaluationWorkflow()
        
        eval_result = await eval_workflow.run_evaluation(gen_result)
        logger.info(f"Evaluation result - Verdict: {eval_result.verdict}")
        
        logger.info("CAB evaluation examples completed successfully!")
        
    except CABEvaluationError as e:
        logger.error(f"CAB evaluation error: {e}")
        logger.error(f"Error code: {e.error_code}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
