#!/usr/bin/env python3
"""Example script to test iterative judge evaluation with repository exploration."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cab_evaluation.core.models import (
    IssueData, CommitInfo, Question, Comment, ConversationMessage,
    JudgeConfig, GenerationResult, VerdictType, SatisfactionStatus
)
from cab_evaluation.agents.agent_factory import AgentFactory
from cab_evaluation.workflows.evaluation_workflow import EvaluationWorkflow
from cab_evaluation.core.config import CABConfig


async def test_iterative_judge():
    """Test the iterative judge functionality."""
    print("🚀 Testing Iterative Judge Agent")
    print("=" * 50)
    
    # Create real issue data (based on actual generation results)
    sample_issue = IssueData(
        id="43",
        language="c++",
        commit_info=CommitInfo(
            repository="https://github.com/esp32-si4732/ats-mini",
            sha="392ce79c18f2cccdb2f9c24985557a47efe2bb5f",
            message="Mode FM under Band CB not selectable",
            author="BrightCGN",
            date="2025-04-09T06:29:21Z"
        ),
        first_question=Question(
            title="Mode FM under Band CB not selectable",
            body="Hello,\n\nI got another issue.\n\nUnder Band CB I'm not able to select Mode FM on my device.\n\nI'd installed V1.09 with rotated display.",
            user="BrightCGN",
            created_at="2025-04-09T06:29:21Z"
        ),
        comments=[
            Comment(
                user="jimjackii",
                body="That is correct. The SI4732 is not capable of FM in the CB band.\n\nRegards, Steffen",
                created_at="2025-04-09T07:01:42Z"
            ),
            Comment(
                user="BrightCGN",
                body="> That is correct. The SI4732 is not capable of FM in the CB band.\n> \n> Regards, Steffen\n\nThanks for the infos :-)",
                created_at="2025-04-09T07:06:09Z"
            )
        ],
        user_satisfaction_condition=[
            "A clear explanation of whether the feature is possible or not",
            "Technical reasoning for why a feature limitation exists"
        ]
    )
    
    # Create sample generation result based on real data
    generation_result = GenerationResult(
        issue_data=sample_issue,
        conversation_history=[
            ConversationMessage(role="user", content="Mode FM under Band CB not selectable\n\nHello,\n\nI got another issue.\n\nUnder Band CB I'm not able to select Mode FM on my device.\n\nI'd installed V1.09 with rotated display."),
            ConversationMessage(role="maintainer", content="Based on my exploration of the repository, I can explain why FM mode is not selectable under CB band on your device. The issue is related to how the ATS-Mini radio firmware restricts certain modes for specific bands...")
        ],
        final_answer="The SI4732 chip is not capable of FM in the CB band. This is a hardware limitation, not a firmware restriction that can be modified.",
        llm_call_counter={"maintainer": 3, "user": 2},
        user_satisfied=True,
        satisfaction_status=SatisfactionStatus.FULLY_SATISFIED,
        satisfaction_reason="The maintainer provided a clear explanation of the feature limitation (FM not available on CB band due to hardware limitation) and sufficient technical reasoning for why this limitation exists."
    )
    
    # Test different judge configurations
    test_configs = [
        {
            "name": "Basic Iterative (3 iterations)",
            "config": JudgeConfig(
                max_iterations=3,
                enable_repository_exploration=False,
                enable_conversation_analysis=True,
                confidence_threshold=0.9
            )
        },
        {
            "name": "Full Iterative with Repository Exploration",
            "config": JudgeConfig(
                max_iterations=5,
                enable_repository_exploration=True,
                enable_conversation_analysis=True,
                exploration_file_limit=10,
                confidence_threshold=0.85,
                early_stopping_enabled=True
            )
        },
        {
            "name": "Conservative (High confidence threshold)",
            "config": JudgeConfig(
                max_iterations=10,
                enable_repository_exploration=True,
                enable_conversation_analysis=True,
                confidence_threshold=0.95,
                early_stopping_enabled=False  # Force all iterations
            )
        }
    ]
    
    # Test each configuration
    for i, test_config in enumerate(test_configs):
        print(f"\n📊 Test {i+1}: {test_config['name']}")
        print("-" * 40)
        
        try:
            # Create evaluation workflow with judge config
            workflow = EvaluationWorkflow(judge_config=test_config["config"])
            
            # Get current working directory as repository path
            repository_path = str(Path.cwd())
            
            # Run iterative evaluation
            start_time = asyncio.get_event_loop().time()
            result = await workflow.run_iterative_evaluation(
                generation_result,
                repository_path,
                agent_model_mapping={"judge": "sonnet37"}
            )
            end_time = asyncio.get_event_loop().time()
            
            # Display results
            print(f"✅ Evaluation completed in {end_time - start_time:.2f} seconds")
            print(f"📊 Final verdict: {result.verdict.value}")
            
            if result.iterative_evaluation:
                iter_eval = result.iterative_evaluation
                print(f"🔄 Iterations completed: {len(iter_eval.iterations)}")
                print(f"⏱️  Total evaluation time: {iter_eval.total_evaluation_time_seconds:.2f}s")
                
                if iter_eval.stopped_early:
                    print(f"⚡ Early stopping: {iter_eval.early_stopping_reason}")
                
                # Show confidence progression
                if iter_eval.confidence_progression:
                    confidence_trend = " → ".join([f"{c:.2f}" for c in iter_eval.confidence_progression])
                    print(f"📈 Confidence progression: {confidence_trend}")
                
                # Show repository exploration results
                if iter_eval.repository_exploration:
                    repo_exp = iter_eval.repository_exploration
                    print(f"📁 Repository exploration: {repo_exp.files_read} files read, {repo_exp.exploration_time_seconds:.2f}s")
                
                # Show final alignment score
                if result.final_alignment_score:
                    score = result.final_alignment_score
                    print(f"📋 Final alignment: {score.satisfied}/{score.total} conditions met ({score.percentage:.1f}%)")
                
                # Show key issues
                if result.key_issues:
                    print(f"⚠️  Key issues: {', '.join(result.key_issues)}")
            
            print(f"🔧 Total LLM calls: {sum(result.llm_calls.values())}")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n🏁 Iterative Judge Testing Complete")


async def compare_single_vs_iterative():
    """Compare single-iteration vs iterative judge results."""
    print("\n🔍 Comparison: Single vs Iterative Judge")
    print("=" * 50)
    
    # Create test issue
    sample_issue = IssueData(
        id="comparison_test",
        language="JavaScript",
        commit_info=CommitInfo(
            repository="js-project",
            sha="def456", 
            message="Fix async issue",
            author="dev",
            date="2024-10-29"
        ),
        first_question=Question(
            title="How to handle async/await properly in JavaScript?",
            body="My async function isn't working as expected. The promises aren't being awaited correctly.",
            user="js-developer",
            created_at="2024-10-29T14:00:00Z"
        ),
        comments=[
            Comment(
                user="maintainer",
                body="You need to use async/await correctly. Make sure your function is marked as async and you're using await for all promises. Example: async function getData() { const result = await fetch('/api/data'); return result.json(); }",
                created_at="2024-10-29T14:05:00Z"
            )
        ],
        user_satisfaction_condition=[
            "Explain the async/await syntax correctly",
            "Provide a working code example",
            "Address common pitfalls"
        ]
    )
    
    generation_result = GenerationResult(
        issue_data=sample_issue,
        final_answer="You need to use async/await correctly. Make sure your function is marked as async and you're using await for all promises. Example: async function getData() { const result = await fetch('/api/data'); return result.json(); }",
        llm_call_counter={"maintainer": 1}
    )
    
    repository_path = str(Path.cwd())
    
    # Test 1: Single iteration (traditional)
    print("\n📍 Single-Iteration Judge (Traditional)")
    single_workflow = EvaluationWorkflow()
    single_start = asyncio.get_event_loop().time()
    single_result = await single_workflow.run_evaluation(generation_result)
    single_time = asyncio.get_event_loop().time() - single_start
    
    print(f"⏱️  Time: {single_time:.2f}s")
    print(f"📊 Verdict: {single_result.verdict.value}")
    print(f"🔧 LLM calls: {sum(single_result.llm_calls.values())}")
    
    # Test 2: Iterative judge
    print("\n🔄 Iterative Judge (Enhanced)")
    iterative_config = JudgeConfig(
        max_iterations=5,
        enable_repository_exploration=True,
        enable_conversation_analysis=True,
        confidence_threshold=0.8
    )
    iterative_workflow = EvaluationWorkflow(judge_config=iterative_config)
    iterative_start = asyncio.get_event_loop().time()
    iterative_result = await iterative_workflow.run_iterative_evaluation(
        generation_result, repository_path
    )
    iterative_time = asyncio.get_event_loop().time() - iterative_start
    
    print(f"⏱️  Time: {iterative_time:.2f}s")
    print(f"📊 Verdict: {iterative_result.verdict.value}")
    print(f"🔧 LLM calls: {sum(iterative_result.llm_calls.values())}")
    if iterative_result.iterative_evaluation:
        print(f"🔄 Iterations: {len(iterative_result.iterative_evaluation.iterations)}")
        if iterative_result.iterative_evaluation.confidence_progression:
            confidence_trend = " → ".join([f"{c:.2f}" for c in iterative_result.iterative_evaluation.confidence_progression])
            print(f"📈 Confidence: {confidence_trend}")
    
    # Comparison
    print("\n📊 Comparison Summary:")
    print(f"Time overhead: {(iterative_time / single_time - 1) * 100:.1f}% increase")
    print(f"LLM call overhead: {(sum(iterative_result.llm_calls.values()) / sum(single_result.llm_calls.values()) - 1) * 100:.1f}% increase")
    print(f"Verdict consistency: {'✅ Same' if single_result.verdict == iterative_result.verdict else '⚠️ Different'}")


if __name__ == "__main__":
    print("🧪 CAB Iterative Judge Agent Testing")
    print("=" * 60)
    
    # Test iterative functionality
    asyncio.run(test_iterative_judge())
    
    # Compare approaches  
    asyncio.run(compare_single_vs_iterative())
    
    print("\n✅ All tests completed!")
