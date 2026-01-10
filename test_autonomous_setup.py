#!/usr/bin/env python3
"""
Test script to validate autonomous setup.
Run this before deploying to production.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")

    try:
        from src.autonomous_orchestrator import AutonomousOrchestrator, OrchestrationConfig
        logger.info("✅ autonomous_orchestrator imported")
    except Exception as e:
        logger.error(f"❌ Failed to import autonomous_orchestrator: {e}")
        return False

    try:
        from src.health_monitor import HealthMonitor, HealthStatus
        logger.info("✅ health_monitor imported")
    except Exception as e:
        logger.error(f"❌ Failed to import health_monitor: {e}")
        return False

    try:
        from src.auto_recovery import AutoRecovery, DatabaseTransaction
        logger.info("✅ auto_recovery imported")
    except Exception as e:
        logger.error(f"❌ Failed to import auto_recovery: {e}")
        return False

    try:
        from src.alerting import Alerting, get_alerting, alert_info
        logger.info("✅ alerting imported")
    except Exception as e:
        logger.error(f"❌ Failed to import alerting: {e}")
        return False

    return True


def test_directory_structure():
    """Test that required directories exist."""
    logger.info("Testing directory structure...")

    required_dirs = ["data", "logs", "output", "cache", "backups", "deployment"]
    all_exist = True

    for dir_name in required_dirs:
        if Path(dir_name).exists():
            logger.info(f"✅ {dir_name}/ exists")
        else:
            logger.warning(f"⚠️  {dir_name}/ missing (will be created automatically)")

    return True


def test_health_monitor():
    """Test health monitoring system."""
    logger.info("Testing health monitor...")

    try:
        from src.health_monitor import HealthMonitor

        monitor = HealthMonitor()
        overall_status, checks = monitor.check_all()

        logger.info(f"Overall health status: {overall_status.value}")

        for check in checks:
            status_symbol = "✅" if check.status.value == "healthy" else "⚠️"
            logger.info(f"  {status_symbol} {check.name}: {check.message}")

        return True

    except Exception as e:
        logger.error(f"❌ Health monitor test failed: {e}")
        return False


def test_auto_recovery():
    """Test auto-recovery system."""
    logger.info("Testing auto-recovery...")

    try:
        from src.auto_recovery import AutoRecovery

        recovery = AutoRecovery()

        # Test database checks
        results = recovery.recover_all_databases()

        for result in results:
            status_symbol = "✅" if result.success else "⚠️"
            logger.info(f"  {status_symbol} {result.message}")

        return True

    except Exception as e:
        logger.error(f"❌ Auto-recovery test failed: {e}")
        return False


def test_alerting():
    """Test alerting system (without sending actual alerts)."""
    logger.info("Testing alerting system...")

    try:
        from src.alerting import Alerting

        alerting = Alerting()

        logger.info(f"  Email alerts: {'enabled' if alerting.email_enabled else 'disabled'}")
        logger.info(f"  Slack alerts: {'enabled' if alerting.slack_enabled else 'disabled'}")
        logger.info(f"  Webhook alerts: {'enabled' if alerting.webhook_enabled else 'disabled'}")

        return True

    except Exception as e:
        logger.error(f"❌ Alerting test failed: {e}")
        return False


def test_orchestrator_config():
    """Test orchestrator configuration."""
    logger.info("Testing orchestrator configuration...")

    try:
        from src.autonomous_orchestrator import OrchestrationConfig

        config = OrchestrationConfig()

        logger.info(f"  Discovery interval: {config.discovery_interval}s")
        logger.info(f"  Processing interval: {config.processing_interval}s")
        logger.info(f"  Max episodes per cycle: {config.max_episodes_per_cycle}")
        logger.info(f"  Max API cost per day: ${config.max_api_cost_per_day}")

        return True

    except Exception as e:
        logger.error(f"❌ Orchestrator config test failed: {e}")
        return False


def test_dependencies():
    """Test that key dependencies are installed."""
    logger.info("Testing dependencies...")

    dependencies = [
        ("openai", "OpenAI"),
        ("feedparser", "feedparser"),
        ("tweepy", "Tweepy"),
        ("dotenv", "python-dotenv"),
        ("psutil", "psutil"),
        ("requests", "requests"),
    ]

    all_installed = True

    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            logger.info(f"✅ {package_name} installed")
        except ImportError:
            logger.error(f"❌ {package_name} NOT installed (run: pip install {package_name})")
            all_installed = False

    return all_installed


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("AUTONOMOUS SETUP VALIDATION")
    logger.info("=" * 60)

    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Health Monitor", test_health_monitor),
        ("Auto-Recovery", test_auto_recovery),
        ("Alerting", test_alerting),
        ("Orchestrator Config", test_orchestrator_config),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info("")
        logger.info(f"--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("")
        logger.info("🎉 All tests passed! System is ready for autonomous deployment.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Configure .env with your API keys")
        logger.info("2. Choose deployment method (see deployment/README.md)")
        logger.info("3. Deploy and monitor!")
        return 0
    else:
        logger.info("")
        logger.info("⚠️  Some tests failed. Please fix issues before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
