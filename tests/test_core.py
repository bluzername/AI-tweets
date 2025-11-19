"""
Core functionality tests - Health monitoring, recovery, alerting.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from health_monitor import HealthMonitor, HealthStatus
from auto_recovery import AutoRecovery, DatabaseTransaction
from alerting import Alerting, AlertLevel, Alert
from rate_limiter import RateLimiter, RateLimitConfig
from cost_tracker import CostTracker, APICost


class TestHealthMonitor:
    """Test health monitoring system."""

    def test_initialization(self):
        """Test health monitor can be initialized."""
        monitor = HealthMonitor()
        assert monitor is not None

    def test_disk_space_check(self):
        """Test disk space health check."""
        monitor = HealthMonitor()
        result = monitor.check_disk_space()

        assert result is not None
        assert result.name == "disk_space"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert result.details is not None
        assert "free_gb" in result.details

    def test_memory_check(self):
        """Test memory health check."""
        monitor = HealthMonitor()
        result = monitor.check_memory()

        assert result is not None
        assert result.name == "memory"
        assert result.details is not None
        assert "percent" in result.details

    def test_cpu_check(self):
        """Test CPU health check."""
        monitor = HealthMonitor()
        result = monitor.check_cpu()

        assert result is not None
        assert result.name == "cpu"
        assert result.details is not None
        assert "percent" in result.details

    def test_directory_check(self):
        """Test directory structure check."""
        monitor = HealthMonitor()
        result = monitor.check_directory_structure()

        assert result is not None
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]

    def test_check_all(self):
        """Test running all health checks."""
        monitor = HealthMonitor()
        overall_status, checks = monitor.check_all()

        assert overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.UNKNOWN]
        assert len(checks) > 0
        assert all(hasattr(check, 'name') for check in checks)
        assert all(hasattr(check, 'status') for check in checks)


class TestAutoRecovery:
    """Test auto-recovery system."""

    def test_initialization(self):
        """Test auto-recovery can be initialized."""
        recovery = AutoRecovery()
        assert recovery is not None

    def test_database_transaction_success(self, temp_db):
        """Test successful database transaction."""
        # Create table
        with DatabaseTransaction(temp_db) as (conn, cursor):
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            cursor.execute("INSERT INTO test (value) VALUES ('test')")

        # Verify data was committed
        import sqlite3
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_database_transaction_rollback(self, temp_db):
        """Test database transaction rollback on error."""
        import sqlite3

        # Create table first
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()

        # Try to insert with error - should rollback
        try:
            with DatabaseTransaction(temp_db) as (conn, cursor):
                cursor.execute("INSERT INTO test (value) VALUES ('test')")
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify data was rolled back
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_retry_with_backoff_success(self):
        """Test retry with backoff on successful call."""
        recovery = AutoRecovery()

        def successful_func():
            return "success"

        success, result = recovery.retry_with_backoff(successful_func, max_attempts=3)

        assert success is True
        assert result == "success"

    def test_retry_with_backoff_failure(self):
        """Test retry with backoff on failed calls."""
        recovery = AutoRecovery()

        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        success, result = recovery.retry_with_backoff(failing_func, max_attempts=3)

        assert success is False
        assert result is None
        assert call_count == 3  # Should have tried 3 times


class TestAlerting:
    """Test alerting system."""

    def test_initialization(self):
        """Test alerting can be initialized."""
        alerting = Alerting()
        assert alerting is not None

    def test_alert_creation(self):
        """Test creating alerts."""
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test Alert",
            message="This is a test",
            timestamp="2025-01-19T12:00:00"
        )

        assert alert.level == AlertLevel.INFO
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            level=AlertLevel.WARNING,
            title="Test",
            message="Message",
            timestamp="2025-01-19T12:00:00",
            details={"key": "value"}
        )

        alert_dict = alert.to_dict()

        assert "level" in alert_dict
        assert alert_dict["level"] == "warning"
        assert "details" in alert_dict
        assert alert_dict["details"]["key"] == "value"

    def test_send_alert(self):
        """Test sending alerts (log channel)."""
        alerting = Alerting()

        # This should not raise an error
        alerting.alert_info("Test", "This is a test alert")
        alerting.alert_warning("Warning", "This is a warning")

        # Check that alerts were recorded
        assert len(alerting.alert_history) > 0


class TestRateLimiter:
    """Test rate limiting system."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter()
        assert limiter is not None
        assert "openai" in limiter.limits
        assert "anthropic" in limiter.limits

    def test_custom_limit(self):
        """Test setting custom rate limit."""
        limiter = RateLimiter()

        custom_config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000
        )

        limiter.set_limit("custom_service", custom_config)

        assert "custom_service" in limiter.limits
        assert limiter.limits["custom_service"].requests_per_minute == 10

    def test_get_status(self):
        """Test getting rate limit status."""
        limiter = RateLimiter()
        status = limiter.get_status("openai")

        assert "service" in status
        assert "limits" in status
        assert "current" in status
        assert "remaining" in status

    def test_execute_with_limit(self):
        """Test executing function with rate limit."""
        limiter = RateLimiter()

        def dummy_func(x):
            return x * 2

        result = limiter.execute_with_limit("openai", dummy_func, 5)

        assert result == 10


class TestCostTracker:
    """Test cost tracking system."""

    def test_initialization(self, temp_db):
        """Test cost tracker initialization."""
        tracker = CostTracker(db_path=temp_db)
        assert tracker is not None

    def test_track_cost(self, temp_db):
        """Test tracking a cost."""
        tracker = CostTracker(db_path=temp_db)

        cost = APICost(
            service="openai",
            operation="completion",
            model="gpt-4",
            tokens_used=1000,
            cost_usd=0.03,
            timestamp="2025-01-19T12:00:00"
        )

        tracker.track_cost(cost)

        # Verify it was recorded
        daily_cost = tracker.get_daily_cost("2025-01-19")
        assert daily_cost == 0.03

    def test_cost_breakdown(self, temp_db):
        """Test getting cost breakdown."""
        tracker = CostTracker(db_path=temp_db)

        # Track multiple costs
        costs = [
            APICost("openai", "completion", "gpt-4", 1000, 0.03, "2025-01-19T12:00:00"),
            APICost("openai", "embedding", "text-embedding-3-small", 5000, 0.0001, "2025-01-19T12:05:00"),
            APICost("anthropic", "completion", "claude-3-sonnet", 800, 0.024, "2025-01-19T12:10:00"),
        ]

        for cost in costs:
            tracker.track_cost(cost)

        breakdown = tracker.get_cost_breakdown(days=7)

        assert "openai" in breakdown
        assert "anthropic" in breakdown
        assert "completion" in breakdown["openai"]

    def test_budget_check(self, temp_db):
        """Test budget checking."""
        tracker = CostTracker(db_path=temp_db)

        cost = APICost("openai", "completion", "gpt-4", 1000, 5.0, "2025-01-19T12:00:00")
        tracker.track_cost(cost)

        budget_status = tracker.check_budget(daily_limit=10.0, monthly_limit=100.0)

        assert "daily" in budget_status
        assert "monthly" in budget_status
        assert budget_status["daily"]["cost"] == 5.0
        assert budget_status["daily"]["exceeded"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
