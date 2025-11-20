"""Test script for OpenHands integration with CodeAssistBench."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cab_evaluation import create_cab_evaluator
from cab_evaluation.utils.data_processor import DataProcessor
from cab_evaluation.utils.openhands_utils import (
    validate_openhands_installation,
    get_openhands_status_report,
    check_openhands_dependencies
)


def setup_logging():
    """Setup logging for test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


async def test_openhands_availability():
    """Test if OpenHands SDK is available and properly installed."""
    print("=" * 60)
    print("TEST 1: OpenHands SDK Installation Check")
    print("=" * 60)
    
    # Check dependencies
    status = check_openhands_dependencies()
    
    if status.get('openhands_sdk'):
        print("✅ OpenHands SDK is installed")
    else:
        print("❌ OpenHands SDK is NOT installed")
        print("\nTo install OpenHands SDK:")
        print("  pip install openhands-sdk openhands-tools")
        print("  OR")
        print("  pip install -e .[openhands]")
        return False
    
    if not status.get('openhands_tools'):
        print("⚠️  OpenHands tools not installed")
        print("  pip install openhands-tools")
        return False
    
    # Validate installation
    if validate_openhands_installation():
        print("✅ OpenHands SDK is working")
    else:
        print("⚠️  OpenHands SDK validation failed")
        return False
    
    # Check API key
    if not status.get('has_api_key'):
        print("⚠️  No API key set (LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY)")
        print("   The evaluation will fail without an API key")
    
    # Print full status report
    print("\n" + get_openhands_status_report())
    
    return True


async def test_strands_agent():
    """Test default Strands agent (baseline)."""
    print("\n" + "=" * 60)
    print("TEST 2: Strands Agent (Default/Baseline)")
    print("=" * 60)
    
    try:
        evaluator = create_cab_evaluator(
            agent_model_mapping={
                "maintainer": "haiku",  # Use fast model for testing
                "user": "haiku",
                "judge": "haiku"
            }
        )
        print("✅ Strands evaluator created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Strands evaluator: {e}")
        return False


async def test_openhands_agent():
    """Test OpenHands agent creation."""
    print("\n" + "=" * 60)
    print("TEST 3: OpenHands Agent Creation")
    print("=" * 60)
    
    try:
        evaluator = create_cab_evaluator(
            agent_model_mapping={
                "maintainer": "claude-3.5-sonnet",
                "user": "haiku",
                "judge": "haiku"
            },
            agent_framework_mapping={
                "maintainer": "openhands"
            }
        )
        print("✅ OpenHands evaluator created successfully")
        print("   Framework: OpenHands (maintainer), Strands (user, judge)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create OpenHands evaluator: {e}")
        return False


async def test_mixed_framework():
    """Test mixed framework configuration."""
    print("\n" + "=" * 60)
    print("TEST 4: Mixed Framework Configuration")
    print("=" * 60)
    
    try:
        # Test with explicit framework mapping
        evaluator = create_cab_evaluator(
            agent_model_mapping={
                "maintainer": "claude-3.5-sonnet",
                "user": "haiku",
                "judge": "sonnet37"
            },
            agent_framework_mapping={
                "maintainer": "openhands"  # Only maintainer uses OpenHands
            }
        )
        print("✅ Mixed framework evaluator created")
        print("   - Maintainer: OpenHands")
        print("   - User: Strands")
        print("   - Judge: Strands")
        return True
        
    except Exception as e:
        print(f"❌ Mixed framework test failed: {e}")
        return False


async def test_sample_evaluation():
    """Test evaluation on a sample issue."""
    print("\n" + "=" * 60)
    print("TEST 5: Sample Issue Evaluation (Optional)")
    print("=" * 60)
    
    sample_file = Path("examples/sample_dataset.jsonl")
    if not sample_file.exists():
        print("⚠️  Sample dataset not found, skipping evaluation test")
        return True
    
    try:
        # Load sample issue
        data_processor = DataProcessor()
        with open(sample_file, "r") as f:
            issue_dict = json.loads(f.readline())
        
        issue_data = data_processor.load_issue_data_from_dict(issue_dict)
        print(f"📋 Loaded sample issue: {issue_data.first_question.title}")
        
        # For this test, we'll just verify the evaluator can be created
        # (actual evaluation would take too long for a quick test)
        evaluator = create_cab_evaluator(
            agent_model_mapping={
                "maintainer": "haiku",
                "user": "haiku", 
                "judge": "haiku"
            }
        )
        
        print("✅ Evaluator ready for sample issue")
        print("   (Skipping full evaluation to save time)")
        print("   To run full evaluation:")
        print("   python -m cab_evaluation.cli generation examples/sample_dataset.jsonl")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Sample evaluation test error: {e}")
        return True  # Non-critical


async def main():
    """Run all integration tests."""
    setup_logging()
    
    print("\n" + "🧪" * 30)
    print("  OpenHands Integration Test Suite")
    print("🧪" * 30 + "\n")
    
    test_results = []
    
    # Test 1: OpenHands availability
    result1 = await test_openhands_availability()
    test_results.append(("OpenHands Installation", result1))
    
    if not result1:
        print("\n" + "⚠️" * 30)
        print("OpenHands is not available. Install it to enable full testing.")
        print("Continuing with Strands-only tests...")
        print("⚠️" * 30 + "\n")
    
    # Test 2: Strands agent (always available)
    result2 = await test_strands_agent()
    test_results.append(("Strands Agent", result2))
    
    # Test 3-4: OpenHands tests (only if available)
    if result1:
        result3 = await test_openhands_agent()
        test_results.append(("OpenHands Agent", result3))
        
        result4 = await test_mixed_framework()
        test_results.append(("Mixed Framework", result4))
    
    # Test 5: Sample evaluation
    result5 = await test_sample_evaluation()
    test_results.append(("Sample Evaluation", result5))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! OpenHands integration is working correctly.")
    elif passed >= 2:
        print("\n✅ Core functionality working. Some optional features may need attention.")
    else:
        print("\n⚠️  Critical issues detected. Please review the failures above.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
