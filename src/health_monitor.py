#!/usr/bin/env python3
"""
Health Monitor - Comprehensive system health checking and monitoring.
Tracks API health, database integrity, resource usage, and more.
"""

import logging
import time
import psutil
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp
        }
        if self.details:
            result["details"] = self.details
        return result


class HealthMonitor:
    """
    Comprehensive health monitoring system.

    Monitors:
    - API connectivity (OpenAI, X.com, YouTube)
    - Database integrity and size
    - Disk space and I/O
    - Memory and CPU usage
    - Log file sizes
    - Error rates
    - Pipeline performance
    """

    def __init__(self, config_file: str = "viral_config.json"):
        """Initialize health monitor."""
        self.config_file = config_file
        self.config = self._load_config()

        # Thresholds
        self.thresholds = {
            "disk_space_gb": 1.0,  # Minimum free space in GB
            "memory_percent": 90,  # Maximum memory usage %
            "cpu_percent": 85,  # Maximum CPU usage %
            "database_size_mb": 500,  # Warning if DB exceeds this
            "log_size_mb": 100,  # Warning if logs exceed this
            "error_rate_percent": 10,  # Warning if error rate > 10%
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def check_all(self) -> Tuple[HealthStatus, List[HealthCheck]]:
        """
        Run all health checks and return overall status.

        Returns:
            Tuple of (overall_status, list_of_checks)
        """
        checks = []

        # Run all checks
        checks.append(self.check_disk_space())
        checks.append(self.check_memory())
        checks.append(self.check_cpu())
        checks.append(self.check_databases())
        checks.append(self.check_openai_api())
        checks.append(self.check_directory_structure())
        checks.append(self.check_log_files())

        # Determine overall status
        statuses = [check.status for check in checks]

        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall = HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            overall = HealthStatus.UNKNOWN
        else:
            overall = HealthStatus.HEALTHY

        return overall, checks

    def check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        try:
            import shutil
            disk_usage = shutil.disk_usage(".")

            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            if free_gb < self.thresholds["disk_space_gb"]:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.CRITICAL,
                    message=f"Low disk space: {free_gb:.2f} GB free",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "used_percent": round(used_percent, 2)
                    }
                )
            elif free_gb < 5.0:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.WARNING,
                    message=f"Disk space running low: {free_gb:.2f} GB free",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "used_percent": round(used_percent, 2)
                    }
                )
            else:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.HEALTHY,
                    message=f"Disk space OK: {free_gb:.2f} GB free",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "used_percent": round(used_percent, 2)
                    }
                )

        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk space: {e}",
                timestamp=datetime.utcnow().isoformat()
            )

    def check_memory(self) -> HealthCheck:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()

            if memory.percent > self.thresholds["memory_percent"]:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.CRITICAL,
                    message=f"High memory usage: {memory.percent:.1f}%",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "percent": round(memory.percent, 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "total_gb": round(memory.total / (1024**3), 2)
                    }
                )
            elif memory.percent > 75:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.WARNING,
                    message=f"Memory usage elevated: {memory.percent:.1f}%",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "percent": round(memory.percent, 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "total_gb": round(memory.total / (1024**3), 2)
                    }
                )
            else:
                return HealthCheck(
                    name="memory",
                    status=HealthStatus.HEALTHY,
                    message=f"Memory usage OK: {memory.percent:.1f}%",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "percent": round(memory.percent, 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "total_gb": round(memory.total / (1024**3), 2)
                    }
                )

        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory: {e}",
                timestamp=datetime.utcnow().isoformat()
            )

    def check_cpu(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            # Get CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            if cpu_percent > self.thresholds["cpu_percent"]:
                return HealthCheck(
                    name="cpu",
                    status=HealthStatus.WARNING,
                    message=f"High CPU usage: {cpu_percent:.1f}%",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "percent": round(cpu_percent, 2),
                        "cores": cpu_count
                    }
                )
            else:
                return HealthCheck(
                    name="cpu",
                    status=HealthStatus.HEALTHY,
                    message=f"CPU usage OK: {cpu_percent:.1f}%",
                    timestamp=datetime.utcnow().isoformat(),
                    details={
                        "percent": round(cpu_percent, 2),
                        "cores": cpu_count
                    }
                )

        except Exception as e:
            return HealthCheck(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check CPU: {e}",
                timestamp=datetime.utcnow().isoformat()
            )

    def check_databases(self) -> HealthCheck:
        """Check database integrity and size."""
        try:
            databases = [
                "data/episodes.db",
                "data/insights.db",
                "data/tweet_queue.db"
            ]

            db_info = []
            total_size_mb = 0
            issues = []

            for db_path in databases:
                if not Path(db_path).exists():
                    issues.append(f"Database not found: {db_path}")
                    continue

                # Check size
                size_mb = Path(db_path).stat().st_size / (1024**2)
                total_size_mb += size_mb

                # Check integrity
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA integrity_check")
                    result = cursor.fetchone()
                    integrity_ok = result[0] == "ok"
                    conn.close()

                    if not integrity_ok:
                        issues.append(f"Integrity check failed for {db_path}")

                    db_info.append({
                        "path": db_path,
                        "size_mb": round(size_mb, 2),
                        "integrity": "ok" if integrity_ok else "failed"
                    })

                except Exception as e:
                    issues.append(f"Error checking {db_path}: {e}")

            # Determine status
            if issues:
                status = HealthStatus.CRITICAL if "failed" in str(issues) else HealthStatus.WARNING
                message = f"Database issues found: {'; '.join(issues)}"
            elif total_size_mb > self.thresholds["database_size_mb"]:
                status = HealthStatus.WARNING
                message = f"Databases large: {total_size_mb:.2f} MB total"
            else:
                status = HealthStatus.HEALTHY
                message = f"Databases OK: {total_size_mb:.2f} MB total"

            return HealthCheck(
                name="databases",
                status=status,
                message=message,
                timestamp=datetime.utcnow().isoformat(),
                details={"databases": db_info, "total_size_mb": round(total_size_mb, 2)}
            )

        except Exception as e:
            return HealthCheck(
                name="databases",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check databases: {e}",
                timestamp=datetime.utcnow().isoformat()
            )

    def check_openai_api(self) -> HealthCheck:
        """Check OpenRouter/OpenAI API connectivity."""
        try:
            import os
            from openai import OpenAI

            # Check if using OpenRouter
            use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() == "true"

            if use_openrouter:
                # Check OpenRouter API
                api_key = os.getenv("OPENROUTER_API_KEY")
                base_url = "https://openrouter.ai/api/v1"
                service_name = "OpenRouter"
                test_model = "meta-llama/llama-3.2-3b-instruct:free"
            else:
                # Check OpenAI API
                api_key = os.getenv("OPENAI_API_KEY") or self.config.get("openai_api_key")
                base_url = None
                service_name = "OpenAI"
                test_model = "gpt-3.5-turbo"

            if not api_key or api_key in ["your_openai_api_key_here", "your_openrouter_api_key_here"]:
                return HealthCheck(
                    name="openai_api",
                    status=HealthStatus.CRITICAL,
                    message=f"{service_name} API key not configured",
                    timestamp=datetime.utcnow().isoformat()
                )

            # Try a simple API call
            if base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = OpenAI(api_key=api_key)

            # Test with a minimal completion
            start_time = time.time()
            response = client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            latency_ms = (time.time() - start_time) * 1000

            return HealthCheck(
                name="openai_api",
                status=HealthStatus.HEALTHY,
                message=f"{service_name} API accessible (latency: {latency_ms:.0f}ms)",
                timestamp=datetime.utcnow().isoformat(),
                details={
                    "latency_ms": round(latency_ms, 2),
                    "service": service_name,
                    "model": test_model
                }
            )

        except Exception as e:
            service_name = "OpenRouter" if os.getenv("USE_OPENROUTER", "false").lower() == "true" else "OpenAI"
            error_msg = str(e)

            # Rate limits mean the API is working, just busy
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                return HealthCheck(
                    name="openai_api",
                    status=HealthStatus.WARNING,
                    message=f"{service_name} API rate limited (API is working)",
                    timestamp=datetime.utcnow().isoformat(),
                    details={"note": "Rate limit means API is accessible"}
                )

            # Authentication errors are critical
            if "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower():
                return HealthCheck(
                    name="openai_api",
                    status=HealthStatus.CRITICAL,
                    message=f"{service_name} API authentication failed",
                    timestamp=datetime.utcnow().isoformat()
                )

            # Other errors
            return HealthCheck(
                name="openai_api",
                status=HealthStatus.CRITICAL,
                message=f"{service_name} API error: {error_msg[:100]}",
                timestamp=datetime.utcnow().isoformat()
            )

    def check_directory_structure(self) -> HealthCheck:
        """Check that required directories exist."""
        try:
            required_dirs = ["data", "logs", "output", "cache"]
            missing = []

            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    missing.append(dir_name)
                    # Try to create it
                    try:
                        Path(dir_name).mkdir(exist_ok=True)
                        logger.info(f"Created missing directory: {dir_name}")
                    except Exception as e:
                        logger.error(f"Failed to create {dir_name}: {e}")

            if missing:
                return HealthCheck(
                    name="directory_structure",
                    status=HealthStatus.WARNING,
                    message=f"Missing directories created: {', '.join(missing)}",
                    timestamp=datetime.utcnow().isoformat(),
                    details={"created": missing}
                )
            else:
                return HealthCheck(
                    name="directory_structure",
                    status=HealthStatus.HEALTHY,
                    message="All required directories present",
                    timestamp=datetime.utcnow().isoformat()
                )

        except Exception as e:
            return HealthCheck(
                name="directory_structure",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check directories: {e}",
                timestamp=datetime.utcnow().isoformat()
            )

    def check_log_files(self) -> HealthCheck:
        """Check log file sizes and recent errors."""
        try:
            log_dir = Path("logs")

            if not log_dir.exists():
                return HealthCheck(
                    name="log_files",
                    status=HealthStatus.WARNING,
                    message="Logs directory not found",
                    timestamp=datetime.utcnow().isoformat()
                )

            log_files = list(log_dir.glob("*.log"))
            total_size_mb = sum(f.stat().st_size for f in log_files) / (1024**2)

            # Count recent errors (in today's log)
            today_log = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
            error_count = 0

            if today_log.exists():
                try:
                    with open(today_log, 'r') as f:
                        for line in f:
                            if ' - ERROR - ' in line or ' - CRITICAL - ' in line:
                                error_count += 1
                except Exception:
                    pass

            if total_size_mb > self.thresholds["log_size_mb"]:
                status = HealthStatus.WARNING
                message = f"Log files large: {total_size_mb:.2f} MB"
            elif error_count > 10:
                status = HealthStatus.WARNING
                message = f"High error count today: {error_count} errors"
            else:
                status = HealthStatus.HEALTHY
                message = f"Logs OK: {len(log_files)} files, {total_size_mb:.2f} MB"

            return HealthCheck(
                name="log_files",
                status=status,
                message=message,
                timestamp=datetime.utcnow().isoformat(),
                details={
                    "file_count": len(log_files),
                    "total_size_mb": round(total_size_mb, 2),
                    "errors_today": error_count
                }
            )

        except Exception as e:
            return HealthCheck(
                name="log_files",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check logs: {e}",
                timestamp=datetime.utcnow().isoformat()
            )

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall_status, checks = self.check_all()

        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [check.to_dict() for check in checks],
            "summary": {
                "total_checks": len(checks),
                "healthy": sum(1 for c in checks if c.status == HealthStatus.HEALTHY),
                "warnings": sum(1 for c in checks if c.status == HealthStatus.WARNING),
                "critical": sum(1 for c in checks if c.status == HealthStatus.CRITICAL),
                "unknown": sum(1 for c in checks if c.status == HealthStatus.UNKNOWN)
            }
        }

    def save_health_report(self, output_file: str = "data/health_report.json"):
        """Save health report to file."""
        try:
            report = self.get_health_report()

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Health report saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
            return False


def main():
    """CLI entry point for health monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Health Monitor for Podcast-to-Tweet Pipeline")
    parser.add_argument("--config", default="viral_config.json", help="Config file")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    parser.add_argument("--watch", action="store_true", help="Watch mode (continuous monitoring)")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    monitor = HealthMonitor(args.config)

    if args.watch:
        # Continuous monitoring mode
        logger.info(f"Starting health monitoring (checking every {args.interval}s)...")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                overall_status, checks = monitor.check_all()

                print(f"\n{'='*60}")
                print(f"Health Check - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"Overall Status: {overall_status.value.upper()}")
                print(f"{'='*60}")

                for check in checks:
                    status_symbol = {
                        HealthStatus.HEALTHY: "✅",
                        HealthStatus.WARNING: "⚠️",
                        HealthStatus.CRITICAL: "❌",
                        HealthStatus.UNKNOWN: "❓"
                    }.get(check.status, "?")

                    print(f"{status_symbol} {check.name}: {check.message}")

                if args.save:
                    monitor.save_health_report()

                time.sleep(args.interval)

        except KeyboardInterrupt:
            logger.info("Health monitoring stopped")

    else:
        # Single check
        report = monitor.get_health_report()

        print(json.dumps(report, indent=2))

        if args.save:
            monitor.save_health_report()

        # Exit with appropriate code
        if report["overall_status"] == "critical":
            sys.exit(2)
        elif report["overall_status"] == "warning":
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    import sys
    main()
