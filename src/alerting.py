#!/usr/bin/env python3
"""
Alerting System - Multi-channel notifications for critical events.
Supports email, Slack, and webhook notifications.
"""

import logging
import smtplib
import os
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert message."""
    level: AlertLevel
    title: str
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp
        }
        if self.details:
            result["details"] = self.details
        return result


class Alerting:
    """
    Multi-channel alerting system.

    Features:
    - Email notifications via SMTP
    - Slack webhooks
    - Custom webhooks
    - Alert throttling (prevent spam)
    - Alert history tracking
    - Channel routing based on severity
    """

    def __init__(self):
        """Initialize alerting system."""

        # Email configuration
        self.email_enabled = self._load_bool_env("ALERT_EMAIL_ENABLED", False)
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.email_from = os.getenv("ALERT_EMAIL_FROM", self.smtp_username)
        self.email_to = os.getenv("ALERT_EMAIL_TO", "").split(",")

        # Slack configuration
        self.slack_enabled = self._load_bool_env("ALERT_SLACK_ENABLED", False)
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

        # Webhook configuration
        self.webhook_enabled = self._load_bool_env("ALERT_WEBHOOK_ENABLED", False)
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")

        # Throttling configuration
        self.throttle_minutes = int(os.getenv("ALERT_THROTTLE_MINUTES", "30"))
        self.last_alerts: Dict[str, datetime] = {}

        # Alert history
        self.alert_history: List[Alert] = []
        self._load_alert_history()

        logger.info("🔔 Alerting system initialized")
        logger.info(f"  Email: {'enabled' if self.email_enabled else 'disabled'}")
        logger.info(f"  Slack: {'enabled' if self.slack_enabled else 'disabled'}")
        logger.info(f"  Webhook: {'enabled' if self.webhook_enabled else 'disabled'}")

    def _load_bool_env(self, key: str, default: bool = False) -> bool:
        """Load boolean from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def _load_alert_history(self):
        """Load alert history from file."""
        try:
            history_file = Path("data/alert_history.json")

            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Convert back to Alert objects (last 100 only)
                    self.alert_history = [
                        Alert(
                            level=AlertLevel(item["level"]),
                            title=item["title"],
                            message=item["message"],
                            timestamp=item["timestamp"],
                            details=item.get("details")
                        )
                        for item in data[-100:]
                    ]

        except Exception as e:
            logger.warning(f"Could not load alert history: {e}")

    def _save_alert_history(self):
        """Save alert history to file."""
        try:
            Path("data").mkdir(exist_ok=True)
            history_file = Path("data/alert_history.json")

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [alert.to_dict() for alert in self.alert_history[-100:]],
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        except Exception as e:
            logger.warning(f"Could not save alert history: {e}")

    def _should_throttle(self, alert_key: str) -> bool:
        """Check if alert should be throttled."""
        if alert_key not in self.last_alerts:
            return False

        time_since_last = datetime.utcnow() - self.last_alerts[alert_key]

        if time_since_last < timedelta(minutes=self.throttle_minutes):
            logger.debug(f"Throttling alert: {alert_key}")
            return True

        return False

    def send_alert(self, level: AlertLevel, title: str, message: str,
                   details: Optional[Dict[str, Any]] = None,
                   channels: Optional[List[AlertChannel]] = None):
        """
        Send an alert through configured channels.

        Args:
            level: Alert severity level
            title: Alert title/subject
            message: Alert message body
            details: Additional details (optional)
            channels: Specific channels to use (defaults to all enabled)
        """

        # Create alert
        alert = Alert(
            level=level,
            title=title,
            message=message,
            timestamp=datetime.utcnow().isoformat(),
            details=details
        )

        # Add to history
        self.alert_history.append(alert)
        self._save_alert_history()

        # Check throttling
        alert_key = f"{level.value}:{title}"
        if self._should_throttle(alert_key):
            logger.debug(f"Alert throttled: {title}")
            return

        # Update throttle tracking
        self.last_alerts[alert_key] = datetime.utcnow()

        # Determine channels
        if channels is None:
            channels = self._get_default_channels(level)

        # Send to each channel
        for channel in channels:
            try:
                if channel == AlertChannel.EMAIL and self.email_enabled:
                    self._send_email(alert)
                elif channel == AlertChannel.SLACK and self.slack_enabled:
                    self._send_slack(alert)
                elif channel == AlertChannel.WEBHOOK and self.webhook_enabled:
                    self._send_webhook(alert)
                elif channel == AlertChannel.LOG:
                    self._log_alert(alert)

            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")

    def _get_default_channels(self, level: AlertLevel) -> List[AlertChannel]:
        """Get default channels based on alert level."""

        # Always log
        channels = [AlertChannel.LOG]

        # Info alerts - just log
        if level == AlertLevel.INFO:
            return channels

        # Warning alerts - log + slack
        if level == AlertLevel.WARNING:
            if self.slack_enabled:
                channels.append(AlertChannel.SLACK)
            return channels

        # Critical alerts - all channels
        if level == AlertLevel.CRITICAL:
            if self.email_enabled:
                channels.append(AlertChannel.EMAIL)
            if self.slack_enabled:
                channels.append(AlertChannel.SLACK)
            if self.webhook_enabled:
                channels.append(AlertChannel.WEBHOOK)

        return channels

    def _send_email(self, alert: Alert):
        """Send alert via email."""
        try:
            if not self.email_to or not self.smtp_username:
                logger.warning("Email not configured, skipping")
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = ", ".join(self.email_to)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"

            # Build body
            body = f"""
Alert Level: {alert.level.value.upper()}
Time: {alert.timestamp}

{alert.message}
"""

            if alert.details:
                body += f"\n\nDetails:\n{json.dumps(alert.details, indent=2)}"

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"📧 Email alert sent: {alert.title}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_slack(self, alert: Alert):
        """Send alert via Slack webhook."""
        try:
            if not self.slack_webhook_url:
                logger.warning("Slack webhook not configured, skipping")
                return

            # Choose emoji and color based on level
            emoji_map = {
                AlertLevel.INFO: ":information_source:",
                AlertLevel.WARNING: ":warning:",
                AlertLevel.CRITICAL: ":rotating_light:"
            }

            color_map = {
                AlertLevel.INFO: "#36a64f",  # green
                AlertLevel.WARNING: "#ff9900",  # orange
                AlertLevel.CRITICAL: "#ff0000"  # red
            }

            # Build Slack message
            payload = {
                "text": f"{emoji_map.get(alert.level, ':bell:')} *{alert.title}*",
                "attachments": [
                    {
                        "color": color_map.get(alert.level, "#dddddd"),
                        "fields": [
                            {
                                "title": "Level",
                                "value": alert.level.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp,
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False
                            }
                        ]
                    }
                ]
            }

            # Add details if present
            if alert.details:
                payload["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": f"```{json.dumps(alert.details, indent=2)}```",
                    "short": False
                })

            # Send to Slack
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"💬 Slack alert sent: {alert.title}")
            else:
                logger.error(f"Slack webhook returned {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_webhook(self, alert: Alert):
        """Send alert via custom webhook."""
        try:
            if not self.webhook_url:
                logger.warning("Webhook URL not configured, skipping")
                return

            # Send alert as JSON
            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code in (200, 201, 202):
                logger.info(f"🔗 Webhook alert sent: {alert.title}")
            else:
                logger.error(f"Webhook returned {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def _log_alert(self, alert: Alert):
        """Log alert to application logs."""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL
        }

        log_level = level_map.get(alert.level, logging.INFO)

        message = f"🔔 ALERT: {alert.title} - {alert.message}"

        if alert.details:
            message += f" | Details: {json.dumps(alert.details)}"

        logger.log(log_level, message)

    def alert_info(self, title: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send an info-level alert."""
        self.send_alert(AlertLevel.INFO, title, message, details)

    def alert_warning(self, title: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send a warning-level alert."""
        self.send_alert(AlertLevel.WARNING, title, message, details)

    def alert_critical(self, title: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send a critical-level alert."""
        self.send_alert(AlertLevel.CRITICAL, title, message, details)

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total = len(self.alert_history)

        if total == 0:
            return {
                "total_alerts": 0,
                "by_level": {},
                "last_alert": None
            }

        # Count by level
        by_level = {}
        for alert in self.alert_history:
            level = alert.level.value
            by_level[level] = by_level.get(level, 0) + 1

        # Recent alerts (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent = [
            a for a in self.alert_history
            if datetime.fromisoformat(a.timestamp) > cutoff
        ]

        return {
            "total_alerts": total,
            "by_level": by_level,
            "recent_24h": len(recent),
            "last_alert": self.alert_history[-1].to_dict() if self.alert_history else None
        }


# Global alerting instance
_alerting_instance = None


def get_alerting() -> Alerting:
    """Get global alerting instance (singleton)."""
    global _alerting_instance

    if _alerting_instance is None:
        _alerting_instance = Alerting()

    return _alerting_instance


def alert_info(title: str, message: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for info alerts."""
    get_alerting().alert_info(title, message, details)


def alert_warning(title: str, message: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for warning alerts."""
    get_alerting().alert_warning(title, message, details)


def alert_critical(title: str, message: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for critical alerts."""
    get_alerting().alert_critical(title, message, details)


def main():
    """CLI entry point for testing alerts."""
    import argparse

    parser = argparse.ArgumentParser(description="Alerting System Test")
    parser.add_argument("--test-email", action="store_true", help="Test email alert")
    parser.add_argument("--test-slack", action="store_true", help="Test Slack alert")
    parser.add_argument("--test-all", action="store_true", help="Test all alert channels")
    parser.add_argument("--stats", action="store_true", help="Show alert statistics")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    alerting = get_alerting()

    if args.stats:
        stats = alerting.get_alert_stats()
        print(json.dumps(stats, indent=2))
        return

    if args.test_email or args.test_all:
        logger.info("Testing email alert...")
        alerting.send_alert(
            AlertLevel.WARNING,
            "Test Email Alert",
            "This is a test email alert from the Podcasts TLDR pipeline.",
            details={"test": True},
            channels=[AlertChannel.EMAIL]
        )

    if args.test_slack or args.test_all:
        logger.info("Testing Slack alert...")
        alerting.send_alert(
            AlertLevel.WARNING,
            "Test Slack Alert",
            "This is a test Slack alert from the Podcasts TLDR pipeline.",
            details={"test": True},
            channels=[AlertChannel.SLACK]
        )

    if args.test_all:
        logger.info("Testing critical alert (all channels)...")
        alerting.alert_critical(
            "Test Critical Alert",
            "This is a test critical alert that should trigger all configured channels.",
            details={"test": True, "severity": "high"}
        )

    logger.info("Alert test complete!")


if __name__ == "__main__":
    main()
