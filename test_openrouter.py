#!/usr/bin/env python3
"""
Test script for OpenRouter integration.
Run this to verify OpenRouter is working correctly.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, 'src')

from multi_model_analyzer import MultiModelAnalyzer, AIModel


def test_openrouter_mode():
    """Test OpenRouter mode initialization and detection."""
    print("=" * 60)
    print("TESTING OPENROUTER INTEGRATION")
    print("=" * 60)

    # Check environment configuration
    print("\n1. Checking Environment Configuration")
    print("-" * 60)

    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")

    print(f"USE_OPENROUTER: {use_openrouter}")
    print(f"OPENROUTER_API_KEY set: {bool(openrouter_key and openrouter_key != 'your_openrouter_api_key_here')}")
    print(f"OPENAI_API_KEY set: {bool(openai_key and openai_key != 'your_openai_api_key_here')}")
    print(f"ANTHROPIC_API_KEY set: {bool(anthropic_key and anthropic_key != 'your_anthropic_api_key_here')}")
    print(f"GOOGLE_API_KEY set: {bool(google_key and google_key != 'your_google_api_key_here')}")

    # Initialize analyzer
    print("\n2. Initializing Multi-Model Analyzer")
    print("-" * 60)

    try:
        analyzer = MultiModelAnalyzer()
        print(f"✅ Analyzer initialized successfully")
        print(f"Mode: {'OpenRouter' if analyzer.use_openrouter else 'Direct APIs'}")
        print(f"Enabled models: {len(analyzer.enabled_models)}")

        for model in analyzer.enabled_models:
            print(f"  - {model.value}")

    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False

    # Check client initialization
    print("\n3. Checking Client Initialization")
    print("-" * 60)

    if analyzer.use_openrouter:
        if analyzer.openrouter_client:
            print("✅ OpenRouter client initialized")
        else:
            print("❌ OpenRouter client NOT initialized")
            print("   Check that OPENROUTER_API_KEY is set correctly")
            return False
    else:
        print(f"OpenAI client: {'✅ initialized' if analyzer.openai_client else '❌ not initialized'}")
        print(f"Anthropic client: {'✅ initialized' if analyzer.anthropic_client else '❌ not initialized'}")
        print(f"Google client: {'✅ initialized' if analyzer.google_client else '❌ not initialized'}")

    # Test model detection
    print("\n4. Testing Model Detection")
    print("-" * 60)

    if len(analyzer.enabled_models) == 0:
        print("❌ No models detected!")
        print("   Make sure you have either:")
        print("   - OPENROUTER_API_KEY set (recommended)")
        print("   - OR individual API keys (OPENAI_API_KEY, etc.)")
        return False

    expected_count = 3 if analyzer.use_openrouter else sum([
        1 if analyzer.openai_client else 0,
        1 if analyzer.anthropic_client else 0,
        1 if analyzer.google_client else 0
    ])

    print(f"Expected models: {expected_count}")
    print(f"Detected models: {len(analyzer.enabled_models)}")

    if len(analyzer.enabled_models) == expected_count:
        print("✅ Model detection working correctly")
    else:
        print("⚠️  Model count mismatch")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if analyzer.use_openrouter:
        if analyzer.openrouter_client and len(analyzer.enabled_models) > 0:
            print("✅ OpenRouter integration is working correctly!")
            print(f"   You have access to {len(analyzer.enabled_models)} AI models with one API key")
            print("\nNext steps:")
            print("1. Run the full test suite: pytest tests/")
            print("2. Try processing a podcast episode")
            print("3. Monitor costs at https://openrouter.ai/activity")
        else:
            print("❌ OpenRouter integration has issues")
            print("\nTroubleshooting:")
            print("1. Check .env file has USE_OPENROUTER=true")
            print("2. Verify OPENROUTER_API_KEY is set correctly")
            print("3. Get your key from https://openrouter.ai/keys")
            return False
    else:
        if len(analyzer.enabled_models) > 0:
            print("✅ Direct API integration is working correctly!")
            print(f"   You have {len(analyzer.enabled_models)} AI model(s) enabled")
            print("\nConsider switching to OpenRouter for:")
            print("- Simpler API key management (1 key instead of 3)")
            print("- Unified cost tracking")
            print("- Access to 100+ models")
        else:
            print("❌ No AI models are enabled")
            print("\nTroubleshooting:")
            print("1. Check .env file has API keys set")
            print("2. Verify keys are correct (no placeholder values)")
            return False

    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    success = test_openrouter_mode()
    sys.exit(0 if success else 1)
