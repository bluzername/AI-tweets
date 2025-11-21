#!/usr/bin/env python3
"""
Auto-Recovery Module - Self-healing capabilities for the pipeline.
Handles failures, corrupted data, API errors, and system issues automatically.
"""

import logging
import sqlite3
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    SKIP = "skip"
    ROLLBACK = "rollback"
    RESET = "reset"
    BACKUP_RESTORE = "backup_restore"
    RECREATE = "recreate"
    ALERT = "alert"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    action_taken: RecoveryAction
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class AutoRecovery:
    """
    Automatic recovery system for pipeline failures.

    Features:
    - Database corruption detection and repair
    - Transaction rollback on failures
    - Automatic retries with exponential backoff
    - Dead letter queue for persistent failures
    - Backup and restore capabilities
    - State recovery after crashes
    """

    def __init__(self, max_retries: int = 3, retry_delay: int = 60):
        """Initialize auto-recovery system."""
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Dead letter queue for persistent failures
        self.dead_letter_queue: List[Dict[str, Any]] = []

        # Recovery history
        self.recovery_history: List[RecoveryResult] = []

        # Setup paths
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)

    def retry_with_backoff(self, func: Callable, *args,
                          max_attempts: Optional[int] = None,
                          **kwargs) -> tuple[bool, Any]:
        """
        Retry a function with exponential backoff.

        Args:
            func: Function to retry
            *args: Positional arguments for func
            max_attempts: Max retry attempts (defaults to self.max_retries)
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (success, result)
        """
        max_attempts = max_attempts or self.max_retries
        delay = self.retry_delay

        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempting {func.__name__} (attempt {attempt + 1}/{max_attempts})")

                result = func(*args, **kwargs)

                logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")

                self._log_recovery(
                    RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.RETRY,
                        message=f"{func.__name__} succeeded after {attempt + 1} attempts",
                        timestamp=datetime.utcnow().isoformat(),
                        details={"attempts": attempt + 1}
                    )
                )

                return True, result

            except Exception as e:
                logger.warning(f"❌ {func.__name__} failed on attempt {attempt + 1}: {e}")

                if attempt < max_attempts - 1:
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"❌ {func.__name__} failed after {max_attempts} attempts")

                    self._log_recovery(
                        RecoveryResult(
                            success=False,
                            action_taken=RecoveryAction.RETRY,
                            message=f"{func.__name__} failed after {max_attempts} attempts",
                            timestamp=datetime.utcnow().isoformat(),
                            details={"attempts": max_attempts, "last_error": str(e)}
                        )
                    )

                    return False, None

    def check_database_integrity(self, db_path: str) -> RecoveryResult:
        """
        Check database integrity and attempt repair if needed.

        Args:
            db_path: Path to SQLite database

        Returns:
            RecoveryResult indicating status
        """
        logger.info(f"Checking database integrity: {db_path}")

        try:
            # Check if file exists
            if not Path(db_path).exists():
                return RecoveryResult(
                    success=False,
                    action_taken=RecoveryAction.ALERT,
                    message=f"Database file not found: {db_path}",
                    timestamp=datetime.utcnow().isoformat()
                )

            # Connect and run integrity check
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()

            if result[0] == "ok":
                conn.close()
                return RecoveryResult(
                    success=True,
                    action_taken=RecoveryAction.SKIP,
                    message=f"Database integrity OK: {db_path}",
                    timestamp=datetime.utcnow().isoformat()
                )

            # Integrity check failed - attempt recovery
            logger.warning(f"⚠️ Database integrity check failed: {result}")

            # Try to recover with backup
            backup_path = self._get_latest_backup(db_path)

            if backup_path:
                logger.info(f"Attempting to restore from backup: {backup_path}")
                conn.close()

                # Backup the corrupted file
                corrupted_backup = f"{db_path}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(db_path, corrupted_backup)

                # Restore from backup
                shutil.copy2(backup_path, db_path)

                # Verify restored database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                verify_result = cursor.fetchone()
                conn.close()

                if verify_result[0] == "ok":
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.BACKUP_RESTORE,
                        message=f"Database restored from backup: {backup_path}",
                        timestamp=datetime.utcnow().isoformat(),
                        details={"backup_path": str(backup_path), "corrupted_saved_to": corrupted_backup}
                    )

            # No backup or restore failed - try vacuum/reindex
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("VACUUM")
                cursor.execute("REINDEX")
                conn.close()

                # Verify
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                verify_result = cursor.fetchone()
                conn.close()

                if verify_result[0] == "ok":
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.RESET,
                        message=f"Database repaired with VACUUM/REINDEX: {db_path}",
                        timestamp=datetime.utcnow().isoformat()
                    )

            except Exception as vacuum_error:
                logger.error(f"VACUUM/REINDEX failed: {vacuum_error}")

            conn.close()

            # Recovery failed
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ALERT,
                message=f"Database recovery failed: {db_path}",
                timestamp=datetime.utcnow().isoformat(),
                details={"integrity_check": result[0]}
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ALERT,
                message=f"Error checking database: {e}",
                timestamp=datetime.utcnow().isoformat()
            )

    def backup_database(self, db_path: str) -> bool:
        """
        Create a backup of a database.

        Args:
            db_path: Path to database to backup

        Returns:
            True if successful
        """
        try:
            if not Path(db_path).exists():
                logger.warning(f"Database not found for backup: {db_path}")
                return False

            # Create backup filename with timestamp
            db_name = Path(db_path).name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{db_name}.{timestamp}.backup"

            # Copy database
            shutil.copy2(db_path, backup_path)

            logger.info(f"✅ Database backed up: {backup_path}")

            # Clean old backups (keep last 7 days)
            self._cleanup_old_backups(db_name, days=7)

            return True

        except Exception as e:
            logger.error(f"Failed to backup database {db_path}: {e}")
            return False

    def _get_latest_backup(self, db_path: str) -> Optional[Path]:
        """Get the latest backup for a database."""
        db_name = Path(db_path).name
        backups = sorted(self.backup_dir.glob(f"{db_name}.*.backup"), reverse=True)

        if backups:
            return backups[0]
        return None

    def _cleanup_old_backups(self, db_name: str, days: int = 7):
        """Remove backups older than specified days."""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            backups = self.backup_dir.glob(f"{db_name}.*.backup")

            for backup in backups:
                if backup.stat().st_mtime < cutoff.timestamp():
                    backup.unlink()
                    logger.info(f"Removed old backup: {backup}")

        except Exception as e:
            logger.warning(f"Error cleaning old backups: {e}")

    def recover_all_databases(self) -> List[RecoveryResult]:
        """Check and recover all databases."""
        databases = [
            "data/episodes.db",
            "data/insights.db",
            "data/tweet_queue.db"
        ]

        results = []

        for db_path in databases:
            # First, create a backup
            self.backup_database(db_path)

            # Then check integrity
            result = self.check_database_integrity(db_path)
            results.append(result)

            self._log_recovery(result)

        return results

    def add_to_dead_letter_queue(self, item: Dict[str, Any], reason: str):
        """
        Add a failed item to dead letter queue for manual review.

        Args:
            item: The failed item (episode, tweet, etc.)
            reason: Reason for failure
        """
        dead_letter_item = {
            "item": item,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": item.get("retry_count", 0)
        }

        self.dead_letter_queue.append(dead_letter_item)

        # Persist to file
        self._save_dead_letter_queue()

        logger.warning(f"⚠️ Added to dead letter queue: {reason}")

    def _save_dead_letter_queue(self):
        """Save dead letter queue to file."""
        try:
            dlq_file = Path("data/dead_letter_queue.json")

            with open(dlq_file, 'w') as f:
                json.dump(self.dead_letter_queue, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save dead letter queue: {e}")

    def _load_dead_letter_queue(self):
        """Load dead letter queue from file."""
        try:
            dlq_file = Path("data/dead_letter_queue.json")

            if dlq_file.exists():
                with open(dlq_file, 'r') as f:
                    self.dead_letter_queue = json.load(f)

                logger.info(f"Loaded {len(self.dead_letter_queue)} items from dead letter queue")

        except Exception as e:
            logger.warning(f"Could not load dead letter queue: {e}")

    def _log_recovery(self, result: RecoveryResult):
        """Log a recovery result."""
        self.recovery_history.append(result)

        # Keep only last 100 recovery results
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]

        # Persist recovery history
        try:
            history_file = Path("data/recovery_history.json")

            with open(history_file, 'w') as f:
                json.dump(
                    [
                        {
                            "success": r.success,
                            "action": r.action_taken.value,
                            "message": r.message,
                            "timestamp": r.timestamp,
                            "details": r.details
                        }
                        for r in self.recovery_history
                    ],
                    f,
                    indent=2
                )

        except Exception as e:
            logger.warning(f"Could not save recovery history: {e}")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        total = len(self.recovery_history)

        if total == 0:
            return {
                "total_recoveries": 0,
                "success_rate": 0.0,
                "common_actions": {},
                "dead_letter_queue_size": len(self.dead_letter_queue)
            }

        successful = sum(1 for r in self.recovery_history if r.success)
        success_rate = (successful / total) * 100

        # Count actions
        action_counts = {}
        for result in self.recovery_history:
            action = result.action_taken.value
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_recoveries": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": round(success_rate, 2),
            "common_actions": action_counts,
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "latest_recovery": self.recovery_history[-1].timestamp if self.recovery_history else None
        }


class DatabaseTransaction:
    """
    Context manager for safe database transactions with auto-rollback.

    Usage:
        with DatabaseTransaction("data/episodes.db") as (conn, cursor):
            cursor.execute("INSERT ...")
            # Auto-commits on success, auto-rolls back on exception
    """

    def __init__(self, db_path: str):
        """Initialize transaction manager."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def __enter__(self):
        """Enter transaction."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self.conn, self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction with commit or rollback."""
        if exc_type is None:
            # Success - commit
            self.conn.commit()
            logger.debug(f"Transaction committed: {self.db_path}")
        else:
            # Exception occurred - rollback
            self.conn.rollback()
            logger.warning(f"Transaction rolled back: {self.db_path} due to {exc_type.__name__}")

        self.conn.close()

        # Don't suppress the exception
        return False


def main():
    """CLI entry point for auto-recovery tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Recovery Tools")
    parser.add_argument("--check-databases", action="store_true", help="Check all databases")
    parser.add_argument("--backup-all", action="store_true", help="Backup all databases")
    parser.add_argument("--stats", action="store_true", help="Show recovery statistics")
    parser.add_argument("--dlq", action="store_true", help="Show dead letter queue")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    recovery = AutoRecovery()

    if args.check_databases:
        logger.info("Checking all databases...")
        results = recovery.recover_all_databases()

        for result in results:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {result.message}")

    elif args.backup_all:
        logger.info("Backing up all databases...")
        databases = ["data/episodes.db", "data/insights.db", "data/tweet_queue.db"]

        for db_path in databases:
            success = recovery.backup_database(db_path)
            status = "✅" if success else "❌"
            logger.info(f"{status} {db_path}")

    elif args.stats:
        stats = recovery.get_recovery_stats()
        print(json.dumps(stats, indent=2))

    elif args.dlq:
        recovery._load_dead_letter_queue()
        print(json.dumps(recovery.dead_letter_queue, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
