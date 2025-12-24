#!/usr/bin/env python3
"""
Web Dashboard - Comprehensive monitoring and control interface.
Real-time view of pipeline status, performance, and manual controls.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import logging
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_monitor import HealthMonitor
from src.performance_tracker import PerformanceTracker
from src.ab_testing_framework import ABTestingFramework
from src.feedback_loop_optimizer import FeedbackLoopOptimizer
from src.optimal_timing import OptimalTimingAnalyzer
import tweepy

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')


# Global instances
health_monitor = None
performance_tracker = None
ab_testing = None
optimizer = None
_initialized = False


def ensure_initialized():
    """Ensure components and template are initialized."""
    global _initialized, health_monitor, performance_tracker, ab_testing, optimizer
    
    if _initialized:
        return

    # Initialize components if needed
    if health_monitor is None:
        init_components()
    
    # Ensure template exists
    create_dashboard_template()
    
    _initialized = True


@app.before_request
def before_request():
    """Run initialization before handling request."""
    ensure_initialized()


def init_components():
    """Initialize dashboard components."""
    global health_monitor, performance_tracker, ab_testing, optimizer

    try:
        # Load config to get credentials
        config_path = Path("viral_config.json")
        twitter_creds = {}
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Get first account credentials
                    accounts = config.get("x_accounts", {})
                    if accounts:
                        # Use the first account found (e.g. podcasts_tldr)
                        first_account = next(iter(accounts.values()))
                        twitter_creds = first_account
                        logger.info("✅ Loaded Twitter credentials from viral_config.json")
            except Exception as e:
                logger.warning(f"Failed to load viral_config.json: {e}")

        health_monitor = HealthMonitor()
        performance_tracker = PerformanceTracker(twitter_credentials=twitter_creds)
        ab_testing = ABTestingFramework()
        optimizer = FeedbackLoopOptimizer(performance_tracker, ab_testing)
        logger.info("✅ Dashboard components initialized")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")


def compute_stats_from_databases():
    """
    Compute orchestrator metrics by querying databases directly.
    This provides a fallback when orchestrator_metrics.json doesn't exist.

    Returns:
        dict: Metrics dictionary with episode and tweet statistics
    """
    metrics = {
        "total_processed": 0,
        "total_tweets_created": 0,
        "total_cost": 0.0,
        "episodes_pending": 0,
        "episodes_failed": 0,
        "episodes_completed": 0,
        "insights_extracted": 0,
        "threads_scheduled": 0,
        "threads_posted": 0,
        "tweets_scheduled": 0,
        "tweets_posted": 0,
        "tweets_draft": 0,
        "tweets_failed": 0,
        "last_updated": datetime.utcnow().isoformat()
    }

    try:
        # Query episodes database
        episodes_db = Path("data/episodes.db")
        if episodes_db.exists():
            conn = sqlite3.connect(str(episodes_db))
            cursor = conn.cursor()

            # Get episode status counts
            cursor.execute("SELECT processing_status, COUNT(*) FROM episodes GROUP BY processing_status")
            status_counts = dict(cursor.fetchall())

            metrics["episodes_completed"] = status_counts.get("completed", 0)
            metrics["episodes_failed"] = status_counts.get("failed", 0)
            metrics["episodes_pending"] = status_counts.get("pending", 0)
            metrics["total_processed"] = metrics["episodes_completed"]

            conn.close()

        # Query insights database
        insights_db = Path("data/insights.db")
        if insights_db.exists():
            conn = sqlite3.connect(str(insights_db))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM insights")
            metrics["insights_extracted"] = cursor.fetchone()[0]
            conn.close()

        # Query thread_queue database (new thread-based system)
        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            cursor = conn.cursor()

            # Get thread status counts
            cursor.execute("SELECT status, COUNT(*) FROM thread_queue GROUP BY status")
            thread_status_counts = dict(cursor.fetchall())

            metrics["threads_scheduled"] = thread_status_counts.get("scheduled", 0)
            metrics["threads_posted"] = thread_status_counts.get("posted", 0)

            # Count total tweets by summing tweets in each thread (each thread has ~6 tweets)
            cursor.execute("SELECT tweets_json FROM thread_queue")
            total_tweets = 0
            for row in cursor.fetchall():
                try:
                    tweets = json.loads(row[0])
                    total_tweets += len(tweets)
                except:
                    total_tweets += 6  # Default assumption

            metrics["total_tweets_created"] = total_tweets

            # Calculate tweets scheduled/posted based on threads
            cursor.execute("SELECT status, tweets_json FROM thread_queue")
            scheduled_tweets = 0
            posted_tweets = 0
            for row in cursor.fetchall():
                try:
                    tweet_count = len(json.loads(row[1]))
                except:
                    tweet_count = 6
                if row[0] == 'scheduled':
                    scheduled_tweets += tweet_count
                elif row[0] == 'posted':
                    posted_tweets += tweet_count

            metrics["tweets_scheduled"] = scheduled_tweets
            metrics["tweets_posted"] = posted_tweets

            conn.close()

        # Also check legacy tweet_queue database
        tweet_queue_db = Path("data/tweet_queue.db")
        if tweet_queue_db.exists():
            conn = sqlite3.connect(str(tweet_queue_db))
            cursor = conn.cursor()

            # Get legacy tweet status counts
            cursor.execute("SELECT status, COUNT(*) FROM tweet_queue GROUP BY status")
            tweet_status_counts = dict(cursor.fetchall())

            # Add to totals (legacy system)
            metrics["tweets_draft"] = tweet_status_counts.get("draft", 0)
            metrics["tweets_failed"] = tweet_status_counts.get("failed", 0)

            conn.close()

    except Exception as e:
        logger.error(f"Error computing stats from databases: {e}")

    return metrics


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get overall system status."""
    try:
        # Health status
        overall_status, checks = health_monitor.check_all()

        # Orchestrator metrics - try file first, fallback to database queries
        metrics_file = Path("data/orchestrator_metrics.json")
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    orchestrator_metrics = json.load(f)
                logger.debug("Loaded orchestrator metrics from JSON file")
            except Exception as e:
                logger.warning(f"Failed to load orchestrator_metrics.json: {e}, falling back to database")
                orchestrator_metrics = compute_stats_from_databases()
        else:
            # File doesn't exist - compute from databases
            logger.info("orchestrator_metrics.json not found, computing from databases")
            orchestrator_metrics = compute_stats_from_databases()

        # Performance stats
        perf_stats = performance_tracker.get_statistics()

        return jsonify({
            'status': 'ok',
            'health': {
                'overall': overall_status.value,
                'checks': [
                    {
                        'name': check.name,
                        'status': check.status.value,
                        'message': check.message
                    }
                    for check in checks
                ]
            },
            'orchestrator': orchestrator_metrics,
            'performance': perf_stats,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sync-metrics', methods=['POST'])
def sync_metrics():
    """
    Manually sync orchestrator metrics from databases.

    This endpoint recomputes all metrics by querying databases directly
    and optionally saves the results to orchestrator_metrics.json.

    Returns:
        JSON with updated metrics and sync timestamp
    """
    try:
        logger.info("Manual metrics sync requested")

        # Compute fresh stats from databases
        stats = compute_stats_from_databases()

        # Optionally save to file for persistence
        metrics_file = Path("data/orchestrator_metrics.json")
        try:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved synced metrics to {metrics_file}")
        except Exception as e:
            logger.warning(f"Could not save metrics to file: {e}")

        return jsonify({
            'status': 'synced',
            'metrics': stats,
            'synced_at': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Failed to sync metrics: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/test-api-connection', methods=['POST'])
def test_api_connection():
    """
    Manually test OpenRouter/OpenAI API connection.

    This is triggered by a button click, not automatic dashboard refresh,
    to avoid unnecessary API costs.

    Returns:
        JSON with API health check result
    """
    try:
        logger.info("Manual API connection test requested")

        # Use the health monitor's API check
        result = health_monitor.check_openai_api()

        return jsonify({
            'status': 'ok',
            'check': {
                'name': result.name,
                'status': result.status.value,
                'message': result.message,
                'timestamp': result.timestamp,
                'details': result.details
            }
        })

    except Exception as e:
        logger.error(f"Failed to test API connection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sync-twitter-metrics', methods=['POST'])
def sync_twitter_metrics():
    """
    Fetch engagement metrics from Twitter for all posted threads.

    This queries the Twitter API for likes, retweets, impressions, etc.
    and stores them in the performance_metrics database.
    """
    try:
        logger.info("Syncing Twitter metrics for posted threads...")

        # Get posted threads with their Twitter IDs
        thread_queue_db = Path("data/thread_queue.db")
        if not thread_queue_db.exists():
            return jsonify({
                'status': 'error',
                'message': 'No thread queue database found'
            }), 404

        conn = sqlite3.connect(str(thread_queue_db))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT thread_id, first_tweet_id, podcast_name, episode_title, tweets_json, posted_time
            FROM thread_queue
            WHERE status = 'posted' AND first_tweet_id IS NOT NULL
        """)

        posted_threads = cursor.fetchall()
        conn.close()

        if not posted_threads:
            return jsonify({
                'status': 'ok',
                'message': 'No posted threads with Twitter IDs found',
                'synced_count': 0
            })

        # Use performance tracker to collect metrics
        synced_count = 0
        failed_count = 0
        results = []

        for thread_id, first_tweet_id, podcast_name, episode_title, tweets_json, posted_time in posted_threads:
            try:
                performance = performance_tracker.collect_metrics(first_tweet_id)

                if performance:
                    # Enrich with context
                    performance.podcast_name = podcast_name
                    performance.episode_id = thread_id

                    # Extract hook pattern from first tweet
                    try:
                        tweets = json.loads(tweets_json)
                        if tweets:
                            first_tweet_content = tweets[0]
                            # Detect hook pattern
                            if first_tweet_content.startswith("🚨"):
                                performance.hook_pattern = "alert_emoji"
                            elif "🧵" in first_tweet_content[:50]:
                                performance.hook_pattern = "thread_emoji"
                            elif first_tweet_content.startswith("🎧"):
                                performance.hook_pattern = "podcast_emoji"
                            else:
                                performance.hook_pattern = "standard"
                    except:
                        pass

                    performance_tracker.save_performance(performance)
                    synced_count += 1

                    results.append({
                        'thread_id': thread_id,
                        'tweet_id': first_tweet_id,
                        'likes': performance.likes,
                        'retweets': performance.retweets,
                        'impressions': performance.impressions,
                        'engagement_rate': round(performance.engagement_rate, 2)
                    })
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"Failed to collect metrics for {first_tweet_id}: {e}")
                failed_count += 1

        logger.info(f"Twitter metrics sync complete: {synced_count} synced, {failed_count} failed")

        return jsonify({
            'status': 'success',
            'message': f'Synced {synced_count} threads, {failed_count} failed',
            'synced_count': synced_count,
            'failed_count': failed_count,
            'results': results
        })

    except Exception as e:
        logger.error(f"Failed to sync Twitter metrics: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/tweet-timeline')
def api_tweet_timeline():
    """
    Get upcoming thread/tweet timeline showing when threads will be posted.

    Now queries the new thread_queue table for thread-based scheduling.
    Falls back to legacy tweet_queue if no threads found.
    """
    try:
        items = []
        now = datetime.utcnow()

        # First, check the new thread_queue database
        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    thread_id,
                    account_name,
                    podcast_name,
                    podcast_handle,
                    episode_title,
                    tweets_json,
                    thumbnail_path,
                    scheduled_time,
                    status
                FROM thread_queue
                WHERE status = 'scheduled'
                ORDER BY scheduled_time ASC
            """)

            for row in cursor.fetchall():
                scheduled_time = datetime.fromisoformat(row['scheduled_time'].replace('Z', '+00:00'))
                time_diff = scheduled_time - now
                hours_until = time_diff.total_seconds() / 3600

                # Parse tweets to get preview
                try:
                    tweets = json.loads(row['tweets_json'])
                    first_tweet = tweets[0] if tweets else "Thread content"
                    tweet_count = len(tweets)
                except:
                    first_tweet = "Thread content"
                    tweet_count = 6

                # Format time until posting
                if hours_until < 0:
                    time_until = "Overdue"
                    status_class = "overdue"
                elif hours_until < 1:
                    minutes = int(time_diff.total_seconds() / 60)
                    time_until = f"in {minutes} minutes"
                    status_class = "imminent"
                elif hours_until < 24:
                    time_until = f"in {int(hours_until)} hours"
                    status_class = "today"
                else:
                    days = int(hours_until / 24)
                    time_until = f"in {days} days"
                    status_class = "future"

                items.append({
                    'tweet_id': row['thread_id'],
                    'item_type': 'thread',
                    'account_name': row['account_name'],
                    'preview': first_tweet[:100] + ('...' if len(first_tweet) > 100 else ''),
                    'scheduled_time': row['scheduled_time'],
                    'scheduled_time_formatted': scheduled_time.strftime('%Y-%m-%d %H:%M UTC'),
                    'time_until': time_until,
                    'status_class': status_class,
                    'podcast_name': row['podcast_name'] or 'Unknown',
                    'podcast_handle': row['podcast_handle'],
                    'episode_title': row['episode_title'],
                    'tweet_count': tweet_count,
                    'has_thumbnail': bool(row['thumbnail_path']),
                    'posting_strategy': 'thread'
                })

            conn.close()

        # Also check legacy tweet_queue for any remaining individual tweets
        tweet_queue_db = Path("data/tweet_queue.db")
        if tweet_queue_db.exists():
            conn = sqlite3.connect(str(tweet_queue_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    tweet_id,
                    account_name,
                    content_json,
                    scheduled_time,
                    status,
                    posting_strategy,
                    podcast_name,
                    episode_id
                FROM tweet_queue
                WHERE status = 'scheduled'
                ORDER BY scheduled_time ASC
                LIMIT 20
            """)

            for row in cursor.fetchall():
                scheduled_time = datetime.fromisoformat(row['scheduled_time'].replace('Z', '+00:00'))
                time_diff = scheduled_time - now
                hours_until = time_diff.total_seconds() / 3600

                try:
                    content = json.loads(row['content_json'])
                    if isinstance(content, dict) and 'content' in content:
                        first_tweet = content['content']
                    else:
                        first_tweet = str(content)[:100]
                except:
                    first_tweet = "Tweet content"

                if hours_until < 0:
                    time_until = "Overdue"
                    status_class = "overdue"
                elif hours_until < 1:
                    minutes = int(time_diff.total_seconds() / 60)
                    time_until = f"in {minutes} minutes"
                    status_class = "imminent"
                elif hours_until < 24:
                    time_until = f"in {int(hours_until)} hours"
                    status_class = "today"
                else:
                    days = int(hours_until / 24)
                    time_until = f"in {days} days"
                    status_class = "future"

                items.append({
                    'tweet_id': row['tweet_id'],
                    'item_type': 'tweet',
                    'account_name': row['account_name'],
                    'preview': first_tweet[:100] + ('...' if len(first_tweet) > 100 else ''),
                    'scheduled_time': row['scheduled_time'],
                    'scheduled_time_formatted': scheduled_time.strftime('%Y-%m-%d %H:%M UTC'),
                    'time_until': time_until,
                    'status_class': status_class,
                    'podcast_name': row['podcast_name'] or 'Unknown',
                    'posting_strategy': row['posting_strategy'],
                    'episode_id': row['episode_id'],
                    'tweet_count': 1
                })

            conn.close()

        # Sort by scheduled time
        items.sort(key=lambda x: x['scheduled_time'])

        return jsonify({
            'status': 'ok',
            'tweets': items,  # Return all items, no limit
            'total': len(items),
            'next_tweet': items[0] if items else None
        })

    except Exception as e:
        logger.error(f"Failed to get tweet timeline: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/post-tweet/<tweet_id>', methods=['POST'])
def api_post_tweet_now(tweet_id):
    """
    Manually post a scheduled thread or tweet immediately.

    Handles both new thread-based posts and legacy individual tweets.
    For threads, posts all tweets as connected replies with media.
    """
    try:
        # Load X.com config
        config_file = Path("viral_config.json")
        if not config_file.exists():
            return jsonify({
                'status': 'error',
                'message': 'Configuration file not found'
            }), 500

        with open(config_file, 'r') as f:
            config = json.load(f)

        x_accounts = config.get('x_accounts', {})

        # First, check if this is a thread in the new thread_queue
        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM thread_queue
                WHERE thread_id = ? AND status = 'scheduled'
            """, (tweet_id,))

            row = cursor.fetchone()

            if row:
                # This is a thread - post as connected replies
                thread_data = dict(row)
                account_name = thread_data['account_name']
                account_config = x_accounts.get(account_name)

                if not account_config:
                    conn.close()
                    return jsonify({
                        'status': 'error',
                        'message': f'Account {account_name} not found in config'
                    }), 404

                # Initialize tweepy client and API
                client = tweepy.Client(
                    bearer_token=account_config.get('bearer_token'),
                    consumer_key=account_config.get('consumer_key'),
                    consumer_secret=account_config.get('consumer_secret'),
                    access_token=account_config.get('access_token'),
                    access_token_secret=account_config.get('access_token_secret'),
                    wait_on_rate_limit=True
                )

                # API v1.1 for media uploads
                auth = tweepy.OAuth1UserHandler(
                    account_config.get('consumer_key'),
                    account_config.get('consumer_secret'),
                    account_config.get('access_token'),
                    account_config.get('access_token_secret')
                )
                api = tweepy.API(auth, timeout=30)  # 30 second timeout

                # Parse tweets
                tweets = json.loads(thread_data['tweets_json'])
                thumbnail_path = thread_data.get('thumbnail_path')
                tweet_ids = []

                import time
                for i, tweet_text in enumerate(tweets):
                    try:
                        if i == 0:
                            # First tweet: include thumbnail if available
                            media = None
                            if thumbnail_path and Path(thumbnail_path).exists():
                                logger.info(f"Uploading media: {thumbnail_path}")
                                # Retry media upload up to 2 times
                                for attempt in range(2):
                                    try:
                                        media = api.media_upload(thumbnail_path)
                                        break
                                    except Exception as upload_err:
                                        if attempt < 1:
                                            logger.warning(f"Media upload attempt {attempt+1} failed, retrying: {upload_err}")
                                            time.sleep(2)
                                        else:
                                            logger.warning(f"Media upload failed after retries, posting without image: {upload_err}")
                                            media = None

                            # Post with or without media
                            if media:
                                response = client.create_tweet(
                                    text=tweet_text,
                                    media_ids=[media.media_id]
                                )
                            else:
                                response = client.create_tweet(text=tweet_text)
                        else:
                            # Reply to previous tweet
                            response = client.create_tweet(
                                text=tweet_text,
                                in_reply_to_tweet_id=tweet_ids[-1]
                            )

                        tweet_ids.append(response.data['id'])
                        logger.info(f"Posted tweet {i+1}/{len(tweets)}: {response.data['id']}")

                        # Small delay between posts
                        if i < len(tweets) - 1:
                            time.sleep(1.5)

                    except Exception as e:
                        logger.error(f"Error posting tweet {i+1} in thread: {e}")
                        # Mark as failed
                        cursor.execute("""
                            UPDATE thread_queue
                            SET status = 'failed',
                                first_tweet_id = ?,
                                updated_at = ?
                            WHERE thread_id = ?
                        """, (tweet_ids[0] if tweet_ids else None, datetime.utcnow().isoformat(), tweet_id))
                        conn.commit()
                        conn.close()
                        return jsonify({
                            'status': 'error',
                            'message': f'Failed to post tweet {i+1}: {str(e)}'
                        }), 500

                # All tweets posted successfully
                cursor.execute("""
                    UPDATE thread_queue
                    SET status = 'posted',
                        first_tweet_id = ?,
                        all_tweet_ids = ?,
                        posted_time = ?,
                        updated_at = ?
                    WHERE thread_id = ?
                """, (tweet_ids[0], json.dumps(tweet_ids), datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), tweet_id))
                conn.commit()
                conn.close()

                logger.info(f"Manually posted thread {tweet_id} - {len(tweet_ids)} tweets: {tweet_ids}")

                return jsonify({
                    'status': 'success',
                    'message': f'Thread posted successfully ({len(tweet_ids)} tweets)',
                    'tweet_id': tweet_id,
                    'x_tweet_id': tweet_ids[0],
                    'all_tweet_ids': tweet_ids,
                    'posted_at': datetime.utcnow().isoformat()
                })

            conn.close()

        # Fall back to legacy tweet_queue
        tweet_queue_db = Path("data/tweet_queue.db")
        if not tweet_queue_db.exists():
            return jsonify({
                'status': 'error',
                'message': 'Tweet not found in any queue'
            }), 404

        conn = sqlite3.connect(str(tweet_queue_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM tweet_queue
            WHERE tweet_id = ? AND status = 'scheduled'
        """, (tweet_id,))

        row = cursor.fetchone()

        if not row:
            conn.close()
            return jsonify({
                'status': 'error',
                'message': f'Tweet {tweet_id} not found or already posted'
            }), 404

        tweet_data = dict(row)
        content_data = json.loads(tweet_data['content_json'])
        account_name = tweet_data['account_name']
        account_config = x_accounts.get(account_name)

        if not account_config:
            conn.close()
            return jsonify({
                'status': 'error',
                'message': f'Account {account_name} not found in config'
            }), 404

        # Initialize tweepy client
        client = tweepy.Client(
            bearer_token=account_config.get('bearer_token'),
            consumer_key=account_config.get('consumer_key'),
            consumer_secret=account_config.get('consumer_secret'),
            access_token=account_config.get('access_token'),
            access_token_secret=account_config.get('access_token_secret'),
            wait_on_rate_limit=True
        )

        # Post the tweet
        tweet_content = content_data.get('content', '')
        response = client.create_tweet(text=tweet_content)

        if response.data and response.data.get('id'):
            cursor.execute("""
                UPDATE tweet_queue
                SET status = 'posted',
                    posted_time = ?,
                    updated_at = ?
                WHERE tweet_id = ?
            """, (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), tweet_id))
            conn.commit()
            conn.close()

            logger.info(f"Manually posted tweet {tweet_id} - X.com ID: {response.data['id']}")

            return jsonify({
                'status': 'success',
                'message': f'Tweet posted successfully',
                'tweet_id': tweet_id,
                'x_tweet_id': response.data['id'],
                'posted_at': datetime.utcnow().isoformat()
            })
        else:
            conn.close()
            return jsonify({
                'status': 'error',
                'message': f"Failed to post tweet: No tweet ID returned"
            }), 500

    except Exception as e:
        logger.error(f"Failed to post tweet {tweet_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/delete-thread/<thread_id>', methods=['DELETE', 'POST'])
def api_delete_thread(thread_id):
    """
    Delete a scheduled thread from the queue.

    This removes the thread from thread_queue and optionally resets
    the episode status to allow reprocessing.
    """
    try:
        deleted_from = []
        episode_id = None

        # Delete from thread_queue
        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            cursor = conn.cursor()

            # Get episode_id before deleting (for potential cleanup)
            cursor.execute("SELECT episode_id, episode_title FROM thread_queue WHERE thread_id = ?", (thread_id,))
            row = cursor.fetchone()
            if row:
                episode_id = row[0]
                episode_title = row[1]

            # Delete the thread
            cursor.execute("DELETE FROM thread_queue WHERE thread_id = ?", (thread_id,))
            if cursor.rowcount > 0:
                deleted_from.append('thread_queue')
                logger.info(f"Deleted thread {thread_id} from thread_queue")

            conn.commit()
            conn.close()

        # Also check legacy tweet_queue
        tweet_queue_db = Path("data/tweet_queue.db")
        if tweet_queue_db.exists():
            conn = sqlite3.connect(str(tweet_queue_db))
            cursor = conn.cursor()

            cursor.execute("DELETE FROM tweet_queue WHERE tweet_id = ?", (thread_id,))
            if cursor.rowcount > 0:
                deleted_from.append('tweet_queue')
                logger.info(f"Deleted {thread_id} from tweet_queue")

            conn.commit()
            conn.close()

        if not deleted_from:
            return jsonify({
                'status': 'error',
                'message': f'Thread {thread_id} not found in any queue'
            }), 404

        return jsonify({
            'status': 'success',
            'message': f'Thread deleted successfully',
            'thread_id': thread_id,
            'episode_id': episode_id,
            'deleted_from': deleted_from
        })

    except Exception as e:
        logger.error(f"Failed to delete thread {thread_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/performance/hooks')
def api_performance_hooks():
    """Get hook pattern performance."""
    try:
        hook_stats = performance_tracker.get_hook_performance(min_samples=3)

        return jsonify({
            'status': 'ok',
            'data': hook_stats
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/performance/time')
def api_performance_time():
    """Get time-of-day performance."""
    try:
        time_stats = performance_tracker.get_time_performance()

        return jsonify({
            'status': 'ok',
            'data': time_stats
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/performance/podcasts')
def api_performance_podcasts():
    """Get podcast ROI."""
    try:
        podcast_roi = performance_tracker.get_podcast_roi()

        return jsonify({
            'status': 'ok',
            'data': podcast_roi
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ab-tests')
def api_ab_tests():
    """Get active A/B tests."""
    try:
        active_tests = ab_testing.get_active_tests()

        tests_data = []
        for test_id in active_tests:
            test_info = ab_testing.get_test_results(test_id)
            tests_data.append(test_info)

        return jsonify({
            'status': 'ok',
            'data': tests_data
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/ab-tests/<test_id>')
def api_ab_test_details(test_id):
    """Get specific A/B test details."""
    try:
        test_info = ab_testing.get_test_results(test_id)

        return jsonify({
            'status': 'ok',
            'data': test_info
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/optimizations')
def api_optimizations():
    """Get optimization recommendations."""
    try:
        recommendations = optimizer.analyze_and_recommend()

        recs_data = [
            {
                'category': r.category,
                'action': r.action,
                'target': r.target,
                'current': r.current_value,
                'recommended': r.recommended_value,
                'confidence': r.confidence,
                'expected_improvement': r.expected_improvement,
                'reasoning': r.reasoning
            }
            for r in recommendations
        ]

        return jsonify({
            'status': 'ok',
            'data': recs_data
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/optimizations/apply', methods=['POST'])
def api_apply_optimization():
    """Apply optimization recommendations."""
    try:
        data = request.get_json()
        confidence_threshold = data.get('confidence_threshold', 0.8)
        max_changes = data.get('max_changes', 3)

        recommendations = optimizer.analyze_and_recommend()
        applied = optimizer.auto_apply_optimizations(
            recommendations,
            confidence_threshold=confidence_threshold,
            max_changes=max_changes
        )

        return jsonify({
            'status': 'ok',
            'applied_count': len(applied),
            'applied': [
                {
                    'category': r.category,
                    'target': r.target,
                    'reasoning': r.reasoning
                }
                for r in applied
            ]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/analytics/overview')
def api_analytics_overview():
    """
    Get analytics overview with posting statistics.

    Returns summary of threads by status, tweets posted, podcasts covered, etc.
    """
    try:
        overview = {
            'threads': {'posted': 0, 'scheduled': 0, 'failed': 0, 'total': 0},
            'tweets_posted': 0,
            'podcasts_covered': 0,
            'first_post_date': None,
            'last_post_date': None,
            'posts_today': 0,
            'posts_this_week': 0,
            'posts_this_month': 0
        }

        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            cursor = conn.cursor()

            # Thread counts by status
            cursor.execute("SELECT status, COUNT(*) FROM thread_queue GROUP BY status")
            for status, count in cursor.fetchall():
                if status in overview['threads']:
                    overview['threads'][status] = count
            overview['threads']['total'] = sum(overview['threads'].values()) - overview['threads']['total']

            # Total tweets posted (count from all_tweet_ids or estimate from tweets_json)
            cursor.execute("""
                SELECT COALESCE(SUM(
                    CASE
                        WHEN all_tweet_ids IS NOT NULL THEN json_array_length(all_tweet_ids)
                        WHEN tweets_json IS NOT NULL THEN json_array_length(tweets_json)
                        ELSE 6
                    END
                ), 0)
                FROM thread_queue WHERE status = 'posted'
            """)
            overview['tweets_posted'] = cursor.fetchone()[0] or 0

            # Unique podcasts
            cursor.execute("SELECT COUNT(DISTINCT podcast_name) FROM thread_queue WHERE podcast_name IS NOT NULL")
            overview['podcasts_covered'] = cursor.fetchone()[0] or 0

            # Date range
            cursor.execute("SELECT MIN(posted_time), MAX(posted_time) FROM thread_queue WHERE status = 'posted'")
            row = cursor.fetchone()
            if row and row[0]:
                overview['first_post_date'] = row[0]
                overview['last_post_date'] = row[1]

            # Posts by time period
            now = datetime.utcnow()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            week_start = (now - timedelta(days=7)).isoformat()
            month_start = (now - timedelta(days=30)).isoformat()

            cursor.execute("SELECT COUNT(*) FROM thread_queue WHERE status = 'posted' AND posted_time >= ?", (today_start,))
            overview['posts_today'] = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM thread_queue WHERE status = 'posted' AND posted_time >= ?", (week_start,))
            overview['posts_this_week'] = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM thread_queue WHERE status = 'posted' AND posted_time >= ?", (month_start,))
            overview['posts_this_month'] = cursor.fetchone()[0] or 0

            conn.close()

        return jsonify({
            'status': 'ok',
            'data': overview
        })

    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/analytics/posted-threads')
def api_analytics_posted_threads():
    """
    Get all posted threads with their tweet IDs for engagement tracking.

    Returns list of posted threads with:
    - thread_id, podcast_name, episode_title
    - posted_time
    - first_tweet_id, all_tweet_ids
    - tweet count
    """
    try:
        threads = []

        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    thread_id,
                    account_name,
                    podcast_name,
                    podcast_handle,
                    episode_title,
                    posted_time,
                    first_tweet_id,
                    all_tweet_ids,
                    tweets_json
                FROM thread_queue
                WHERE status = 'posted'
                ORDER BY posted_time DESC
            """)

            for row in cursor.fetchall():
                thread_data = dict(row)

                # Parse tweet IDs
                if thread_data['all_tweet_ids']:
                    try:
                        thread_data['all_tweet_ids'] = json.loads(thread_data['all_tweet_ids'])
                    except:
                        thread_data['all_tweet_ids'] = []
                else:
                    thread_data['all_tweet_ids'] = []

                # Count tweets
                if thread_data['tweets_json']:
                    try:
                        tweets = json.loads(thread_data['tweets_json'])
                        thread_data['tweet_count'] = len(tweets)
                    except:
                        thread_data['tweet_count'] = 6
                else:
                    thread_data['tweet_count'] = 6

                # Generate X.com URL for first tweet
                if thread_data['first_tweet_id']:
                    thread_data['thread_url'] = f"https://x.com/i/status/{thread_data['first_tweet_id']}"

                # Remove raw JSON from response
                del thread_data['tweets_json']

                threads.append(thread_data)

            conn.close()

        return jsonify({
            'status': 'ok',
            'data': threads,
            'total': len(threads)
        })

    except Exception as e:
        logger.error(f"Error getting posted threads: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/analytics/timeline-chart')
def api_analytics_timeline_chart():
    """
    Get posting timeline data for charts.

    Returns posts grouped by day for visualization.
    """
    try:
        days = int(request.args.get('days', 30))

        timeline_data = []

        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            cursor = conn.cursor()

            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    DATE(posted_time) as post_date,
                    COUNT(*) as thread_count,
                    SUM(
                        CASE
                            WHEN all_tweet_ids IS NOT NULL THEN json_array_length(all_tweet_ids)
                            WHEN tweets_json IS NOT NULL THEN json_array_length(tweets_json)
                            ELSE 6
                        END
                    ) as tweet_count
                FROM thread_queue
                WHERE status = 'posted' AND posted_time >= ?
                GROUP BY DATE(posted_time)
                ORDER BY post_date ASC
            """, (start_date,))

            for row in cursor.fetchall():
                timeline_data.append({
                    'date': row[0],
                    'threads': row[1],
                    'tweets': row[2] or 0
                })

            conn.close()

        return jsonify({
            'status': 'ok',
            'data': timeline_data,
            'period_days': days
        })

    except Exception as e:
        logger.error(f"Error getting timeline chart: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/analytics/followers')
def api_analytics_followers():
    """
    Get follower count history for tracking growth.

    Returns daily follower counts for visualization.
    """
    try:
        follower_data = []

        # Create/use analytics database for follower tracking
        analytics_db = Path("data/analytics.db")
        analytics_db.parent.mkdir(exist_ok=True)

        conn = sqlite3.connect(str(analytics_db))
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS follower_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT NOT NULL,
                follower_count INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                source TEXT DEFAULT 'manual'
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_follower_date ON follower_history (account_name, recorded_at)")
        conn.commit()

        # Get follower history
        days = int(request.args.get('days', 30))
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT account_name, follower_count, recorded_at, source
            FROM follower_history
            WHERE recorded_at >= ?
            ORDER BY recorded_at ASC
        """, (start_date,))

        for row in cursor.fetchall():
            follower_data.append({
                'account': row[0],
                'count': row[1],
                'date': row[2],
                'source': row[3]
            })

        # Get latest count per account
        cursor.execute("""
            SELECT account_name, follower_count, recorded_at
            FROM follower_history
            WHERE (account_name, recorded_at) IN (
                SELECT account_name, MAX(recorded_at)
                FROM follower_history
                GROUP BY account_name
            )
        """)

        current_counts = {}
        for row in cursor.fetchall():
            current_counts[row[0]] = {
                'count': row[1],
                'as_of': row[2]
            }

        conn.close()

        return jsonify({
            'status': 'ok',
            'data': follower_data,
            'current': current_counts,
            'period_days': days
        })

    except Exception as e:
        logger.error(f"Error getting follower history: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/analytics/followers/record', methods=['POST'])
def api_analytics_record_followers():
    """
    Record a follower count snapshot.

    POST body: { "account_name": "podcasts_tldr", "follower_count": 1234 }

    Can be called manually or by a scheduled job that scrapes the count.
    """
    try:
        data = request.get_json()
        account_name = data.get('account_name', 'podcasts_tldr')
        follower_count = data.get('follower_count')
        source = data.get('source', 'manual')

        if follower_count is None:
            return jsonify({
                'status': 'error',
                'message': 'follower_count is required'
            }), 400

        analytics_db = Path("data/analytics.db")
        analytics_db.parent.mkdir(exist_ok=True)

        conn = sqlite3.connect(str(analytics_db))
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS follower_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT NOT NULL,
                follower_count INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                source TEXT DEFAULT 'manual'
            )
        """)

        # Insert new record
        cursor.execute("""
            INSERT INTO follower_history (account_name, follower_count, recorded_at, source)
            VALUES (?, ?, ?, ?)
        """, (account_name, follower_count, datetime.utcnow().isoformat(), source))

        conn.commit()
        conn.close()

        logger.info(f"Recorded follower count: {account_name} = {follower_count}")

        return jsonify({
            'status': 'success',
            'message': f'Recorded {follower_count} followers for {account_name}',
            'account': account_name,
            'count': follower_count,
            'recorded_at': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Error recording follower count: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/analytics/by-podcast')
def api_analytics_by_podcast():
    """
    Get posting statistics grouped by podcast.

    Returns threads and tweets per podcast.
    """
    try:
        podcast_stats = []

        thread_queue_db = Path("data/thread_queue.db")
        if thread_queue_db.exists():
            conn = sqlite3.connect(str(thread_queue_db))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COALESCE(podcast_name, 'Unknown') as podcast,
                    podcast_handle,
                    COUNT(*) as thread_count,
                    SUM(CASE WHEN status = 'posted' THEN 1 ELSE 0 END) as posted_count,
                    SUM(CASE WHEN status = 'scheduled' THEN 1 ELSE 0 END) as scheduled_count,
                    MIN(CASE WHEN status = 'posted' THEN posted_time END) as first_post,
                    MAX(CASE WHEN status = 'posted' THEN posted_time END) as last_post
                FROM thread_queue
                GROUP BY COALESCE(podcast_name, 'Unknown'), podcast_handle
                ORDER BY thread_count DESC
            """)

            for row in cursor.fetchall():
                podcast_stats.append({
                    'podcast_name': row[0],
                    'podcast_handle': row[1],
                    'total_threads': row[2],
                    'posted': row[3] or 0,
                    'scheduled': row[4] or 0,
                    'first_post': row[5],
                    'last_post': row[6]
                })

            conn.close()

        return jsonify({
            'status': 'ok',
            'data': podcast_stats,
            'total_podcasts': len(podcast_stats)
        })

    except Exception as e:
        logger.error(f"Error getting podcast analytics: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/analytics/timing-report')
def api_analytics_timing_report():
    """
    Get optimal posting time analysis report.

    Returns analysis of historical posting performance and optimal times.
    """
    try:
        analyzer = OptimalTimingAnalyzer()
        report = analyzer.get_timing_report()

        # Get next optimal time
        next_optimal = analyzer.get_next_optimal_time()

        return jsonify({
            'status': 'ok',
            'data': {
                'total_posts_analyzed': report['total_posts_analyzed'],
                'recommendation': report['recommendation'],
                'optimal_weekday': report['optimal_weekday'],
                'optimal_weekend': report['optimal_weekend'],
                'best_performing_hours': report['best_performing_hours'],
                'worst_performing_hours': report['worst_performing_hours'],
                'currently_scheduled_hours': report['currently_scheduled_hours'],
                'next_optimal_time': next_optimal.isoformat() if next_optimal else None
            }
        })

    except Exception as e:
        logger.error(f"Error getting timing report: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    try:
        overall_status, _ = health_monitor.check_all()

        return jsonify({
            'status': 'healthy' if overall_status.value == 'healthy' else 'degraded',
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503


@app.route('/api/factcheck/overview')
def api_factcheck_overview():
    """Get fact-check pipeline overview statistics."""
    try:
        db_path = Path("data/factcheck_results.db")
        if not db_path.exists():
            return jsonify({
                'status': 'success',
                'data': {
                    'total_runs': 0,
                    'total_claims': 0,
                    'false_claims': 0,
                    'misleading_claims': 0,
                    'true_claims': 0,
                    'debunk_threads_scheduled': 0,
                    'recent_runs': []
                }
            })

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get summary stats
        cursor.execute("SELECT COUNT(*) FROM factcheck_runs")
        total_runs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM factcheck_claims")
        total_claims = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM factcheck_claims WHERE verdict = 'false'")
        false_claims = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM factcheck_claims WHERE verdict = 'misleading'")
        misleading_claims = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM factcheck_claims WHERE verdict = 'true'")
        true_claims = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM factcheck_runs WHERE debunk_thread_scheduled = 1")
        debunk_scheduled = cursor.fetchone()[0]

        # Get recent runs
        cursor.execute('''
            SELECT podcast_name, episode_title, total_claims_extracted,
                   false_claims_found, misleading_claims_found, debunk_thread_scheduled,
                   created_at
            FROM factcheck_runs
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        recent_runs = []
        for row in cursor.fetchall():
            recent_runs.append({
                'podcast_name': row[0],
                'episode_title': row[1],
                'total_claims': row[2],
                'false_claims': row[3],
                'misleading_claims': row[4],
                'debunk_scheduled': row[5] == 1,
                'created_at': row[6]
            })

        conn.close()

        return jsonify({
            'status': 'success',
            'data': {
                'total_runs': total_runs,
                'total_claims': total_claims,
                'false_claims': false_claims,
                'misleading_claims': misleading_claims,
                'true_claims': true_claims,
                'debunk_threads_scheduled': debunk_scheduled,
                'recent_runs': recent_runs
            }
        })

    except Exception as e:
        logger.error(f"Error getting factcheck overview: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/factcheck/claims')
def api_factcheck_claims():
    """Get detailed fact-check claims with filtering."""
    try:
        verdict_filter = request.args.get('verdict', None)  # false, misleading, true, unverified
        limit = min(int(request.args.get('limit', 50)), 200)

        db_path = Path("data/factcheck_results.db")
        if not db_path.exists():
            return jsonify({'status': 'success', 'data': {'claims': []}})

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        query = '''
            SELECT c.claim_text, c.verdict, c.correction, c.source_name, c.source_url,
                   c.confidence, c.method_used, c.created_at, r.podcast_name, r.episode_title
            FROM factcheck_claims c
            JOIN factcheck_runs r ON c.run_id = r.id
        '''

        params = []
        if verdict_filter:
            query += " WHERE c.verdict = ?"
            params.append(verdict_filter)

        query += " ORDER BY c.created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        claims = []
        for row in cursor.fetchall():
            claims.append({
                'claim_text': row[0],
                'verdict': row[1],
                'correction': row[2],
                'source_name': row[3],
                'source_url': row[4],
                'confidence': row[5],
                'method_used': row[6],
                'created_at': row[7],
                'podcast_name': row[8],
                'episode_title': row[9]
            })

        conn.close()

        return jsonify({'status': 'success', 'data': {'claims': claims}})

    except Exception as e:
        logger.error(f"Error getting factcheck claims: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/factcheck/debunk-threads')
def api_factcheck_debunk_threads():
    """Get PodDebunker threads from queue."""
    try:
        db_path = Path("data/thread_queue.db")
        if not db_path.exists():
            return jsonify({'status': 'success', 'data': {'threads': []}})

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT thread_id, podcast_name, episode_title, status,
                   scheduled_time, posted_time, created_at
            FROM thread_queue
            WHERE account_name = 'poddebunker'
            ORDER BY created_at DESC
            LIMIT 20
        ''')

        threads = []
        for row in cursor.fetchall():
            threads.append({
                'thread_id': row[0],
                'podcast_name': row[1],
                'episode_title': row[2],
                'status': row[3],
                'scheduled_time': row[4],
                'posted_time': row[5],
                'created_at': row[6]
            })

        conn.close()

        return jsonify({'status': 'success', 'data': {'threads': threads}})

    except Exception as e:
        logger.error(f"Error getting debunk threads: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Template creation helper
def create_dashboard_template():
    """Create the dashboard HTML template."""

    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)

    template_path = template_dir / 'dashboard.html'

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Podcasts TLDR - Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            opacity: 0.9;
            font-size: 1.1em;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #334155;
        }

        .card h2 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #a78bfa;
        }

        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-healthy { background: #10b981; color: white; }
        .status-warning { background: #f59e0b; color: white; }
        .status-critical { background: #ef4444; color: white; }

        .metric {
            margin: 15px 0;
            padding: 15px;
            background: #0f172a;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .metric-label {
            font-size: 0.9em;
            color: #94a3b8;
            margin-bottom: 5px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: 700;
            color: #a78bfa;
        }

        .metric-small {
            font-size: 1em;
        }

        .health-check {
            padding: 10px;
            margin: 8px 0;
            background: #0f172a;
            border-radius: 6px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .health-icon {
            font-size: 1.5em;
        }

        .recommendation {
            padding: 15px;
            margin: 10px 0;
            background: #0f172a;
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
        }

        .recommendation h3 {
            color: #fbbf24;
            margin-bottom: 8px;
            font-size: 1.1em;
        }

        .confidence-bar {
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #3b82f6);
            transition: width 0.3s ease;
        }

        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #94a3b8;
        }

        .chart-container {
            margin-top: 20px;
            height: 200px;
            background: #0f172a;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            align-items: flex-end;
            gap: 5px;
        }

        .chart-bar {
            flex: 1;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px 4px 0 0;
            transition: height 0.3s ease;
            position: relative;
        }

        .chart-label {
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.7em;
            color: #94a3b8;
        }

        /* Tab Navigation */
        .tab-btn {
            background: rgba(255,255,255,0.1);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.95em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tab-btn:hover {
            background: rgba(255,255,255,0.2);
        }

        .tab-btn.active {
            background: white;
            color: #764ba2;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Analytics specific styles */
        .big-stat {
            text-align: center;
            padding: 1.5rem;
        }

        .big-stat .value {
            font-size: 3em;
            font-weight: 700;
            color: #a78bfa;
        }

        .big-stat .label {
            font-size: 0.9em;
            color: #94a3b8;
            margin-top: 0.5rem;
        }

        .thread-item {
            background: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 3px solid #667eea;
            transition: transform 0.2s;
        }

        .thread-item:hover {
            transform: translateX(5px);
        }

        .thread-item a {
            color: #a78bfa;
            text-decoration: none;
        }

        .thread-item a:hover {
            text-decoration: underline;
        }

        .timeline-bar-chart {
            display: flex;
            align-items: flex-end;
            gap: 4px;
            height: 200px;
            padding: 1rem;
            background: #0f172a;
            border-radius: 8px;
        }

        .timeline-bar {
            flex: 1;
            min-width: 20px;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px 4px 0 0;
            position: relative;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .timeline-bar:hover {
            opacity: 0.8;
        }

        .timeline-bar .tooltip {
            display: none;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #1e293b;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.8em;
            white-space: nowrap;
            z-index: 10;
        }

        .timeline-bar:hover .tooltip {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎙️ Podcasts TLDR Dashboard</h1>
            <p class="subtitle">Autonomous Content Machine - Real-time Monitoring & Control</p>
            <div class="tab-nav" style="margin-top: 1.5rem; display: flex; gap: 0.5rem;">
                <button class="tab-btn active" onclick="showTab('dashboard')">📊 Dashboard</button>
                <button class="tab-btn" onclick="showTab('factcheck')">🔍 Fact-Check</button>
                <button class="tab-btn" onclick="showTab('analytics')">📈 Analytics</button>
                <button class="tab-btn" onclick="showTab('threads')">🧵 Posted Threads</button>
            </div>
        </header>

        <!-- Dashboard Tab -->
        <div id="tab-dashboard" class="tab-content active">
        <div class="grid">
            <div class="card">
                <h2>📊 System Status</h2>
                <div id="system-status">
                    <div class="loading">Loading...</div>
                </div>
            </div>

            <div class="card">
                <h2>📈 Performance Metrics</h2>
                <div id="performance-metrics">
                    <div class="loading">Loading...</div>
                </div>
                <button onclick="syncTwitterMetrics()" style="margin-top: 15px; background: #3b82f6;">
                    🔄 Sync Twitter Metrics
                </button>
            </div>

            <div class="card">
                <h2>🔄 Orchestrator Stats</h2>
                <div id="orchestrator-stats">
                    <div class="loading">Loading...</div>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>🏥 Health Checks</h2>
                <div id="health-checks">
                    <div class="loading">Loading...</div>
                </div>
                <button onclick="testApiConnection()" style="margin-top: 15px; background: #10b981;">
                    🔌 Test API Connection
                </button>
                <div id="api-test-result" style="margin-top: 10px; display: none;"></div>
            </div>

            <div class="card">
                <h2>🎣 Hook Performance</h2>
                <div id="hook-performance">
                    <div class="loading">Loading...</div>
                </div>
            </div>

            <div class="card">
                <h2>⏰ Time-of-Day Performance</h2>
                <div id="time-performance">
                    <div class="loading">Loading...</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🧪 Active A/B Tests</h2>
            <div id="ab-tests">
                <div class="loading">Loading...</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 2rem;">
            <h2>📅 Upcoming Tweet Timeline</h2>
            <div id="tweet-timeline">
                <div class="loading">Loading timeline...</div>
            </div>
        </div>

        <div class="card">
            <h2>💡 Optimization Recommendations</h2>
            <div id="recommendations">
                <div class="loading">Loading...</div>
            </div>
            <button onclick="applyOptimizations()" style="margin-top: 15px;">
                🚀 Auto-Apply High-Confidence Optimizations
            </button>
        </div>
        </div> <!-- End Dashboard Tab -->

        <!-- Fact-Check Tab -->
        <div id="tab-factcheck" class="tab-content" style="display: none;">
            <div class="grid">
                <div class="card">
                    <h2>🔍 Fact-Check Overview</h2>
                    <div id="factcheck-overview">
                        <div class="loading">Loading...</div>
                    </div>
                </div>

                <div class="card">
                    <h2>🎯 PodDebunker Threads</h2>
                    <div id="debunk-threads">
                        <div class="loading">Loading...</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>📋 Recent Fact-Check Runs</h2>
                <div id="factcheck-runs">
                    <div class="loading">Loading...</div>
                </div>
            </div>

            <div class="card">
                <h2>⚠️ False/Misleading Claims Found</h2>
                <div id="factcheck-claims">
                    <div class="loading">Loading...</div>
                </div>
            </div>
        </div> <!-- End Fact-Check Tab -->

        <!-- Analytics Tab -->
        <div id="tab-analytics" class="tab-content" style="display: none;">
            <div class="grid">
                <div class="card">
                    <h2>📊 Posting Overview</h2>
                    <div id="analytics-overview">
                        <div class="loading">Loading analytics...</div>
                    </div>
                </div>

                <div class="card">
                    <h2>📅 Posts by Period</h2>
                    <div id="analytics-period">
                        <div class="loading">Loading...</div>
                    </div>
                </div>

                <div class="card">
                    <h2>🎙️ Podcasts Covered</h2>
                    <div id="analytics-podcasts">
                        <div class="loading">Loading...</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>📈 Posting Timeline (Last 30 Days)</h2>
                <div id="analytics-timeline-chart">
                    <div class="loading">Loading chart...</div>
                </div>
            </div>

            <div class="card">
                <h2>👥 Follower Growth</h2>
                <div class="grid" style="grid-template-columns: 1fr 2fr; gap: 1.5rem;">
                    <div>
                        <h3 style="font-size: 1em; color: #94a3b8; margin-bottom: 1rem;">Record Follower Count</h3>
                        <div style="background: #0f172a; padding: 1rem; border-radius: 8px;">
                            <input type="number" id="follower-count-input" placeholder="Enter follower count"
                                   style="width: 100%; padding: 0.75rem; border: 1px solid #334155; border-radius: 6px; background: #1e293b; color: white; margin-bottom: 0.75rem;">
                            <button onclick="recordFollowerCount()" style="width: 100%; padding: 0.75rem; background: #10b981;">
                                📝 Record Today's Count
                            </button>
                        </div>
                        <div id="follower-current" style="margin-top: 1rem;">
                            <div class="loading">Loading...</div>
                        </div>
                    </div>
                    <div>
                        <h3 style="font-size: 1em; color: #94a3b8; margin-bottom: 1rem;">Growth History</h3>
                        <div id="follower-chart">
                            <div class="loading">Loading follower data...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div> <!-- End Analytics Tab -->

        <!-- Posted Threads Tab -->
        <div id="tab-threads" class="tab-content" style="display: none;">
            <div class="card">
                <h2>🧵 Posted Threads</h2>
                <p style="color: #94a3b8; margin-bottom: 1rem;">Click on a thread to view it on X.com. Tweet IDs are tracked for engagement analytics.</p>
                <div id="posted-threads-list">
                    <div class="loading">Loading posted threads...</div>
                </div>
            </div>
        </div> <!-- End Threads Tab -->
    </div>

    <script>
        // Fetch and update all data
        async function updateDashboard() {
            try {
                // System status
                const status = await fetch('/api/status').then(r => r.json());
                updateSystemStatus(status);
                updatePerformanceMetrics(status.performance);
                updateOrchestratorStats(status.orchestrator);
                updateHealthChecks(status.health.checks);

                // Hook performance
                const hooks = await fetch('/api/performance/hooks').then(r => r.json());
                updateHookPerformance(hooks.data);

                // Time performance
                const timePerf = await fetch('/api/performance/time').then(r => r.json());
                updateTimePerformance(timePerf.data);

                // A/B tests
                const tests = await fetch('/api/ab-tests').then(r => r.json());
                updateABTests(tests.data);

                // Recommendations
                const recs = await fetch('/api/optimizations').then(r => r.json());
                updateRecommendations(recs.data);

                // Tweet timeline
                const timeline = await fetch('/api/tweet-timeline').then(r => r.json());
                updateTweetTimeline(timeline);

            } catch (error) {
                console.error('Dashboard update failed:', error);
            }
        }

        function updateSystemStatus(status) {
            const statusClass = `status-${status.health.overall}`;
            document.getElementById('system-status').innerHTML = `
                <div class="metric">
                    <div class="metric-label">Overall Health</div>
                    <div><span class="status-badge ${statusClass}">${status.health.overall}</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Last Updated</div>
                    <div class="metric-value metric-small">${new Date(status.timestamp).toLocaleString()}</div>
                </div>
            `;
        }

        function updatePerformanceMetrics(perf) {
            const hasData = perf.total_tweets > 0;
            if (hasData) {
                document.getElementById('performance-metrics').innerHTML = `
                    <div class="metric">
                        <div class="metric-label">Total Tweets Tracked</div>
                        <div class="metric-value">${perf.total_tweets || 0}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Engagement Rate</div>
                        <div class="metric-value metric-small">${perf.avg_engagement_rate || 0}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg Likes / Retweets</div>
                        <div class="metric-value metric-small">${perf.avg_likes || 0} / ${perf.avg_retweets || 0}</div>
                    </div>
                `;
            } else {
                document.getElementById('performance-metrics').innerHTML = `
                    <p style="color: #94a3b8; text-align: center; padding: 1rem;">
                        No engagement data yet.<br>
                        Click "Sync Twitter Metrics" to fetch engagement data for posted tweets.
                    </p>
                `;
            }
        }

        function updateOrchestratorStats(stats) {
            document.getElementById('orchestrator-stats').innerHTML = `
                <div class="metric">
                    <div class="metric-label">Episodes</div>
                    <div class="metric-value metric-small">
                        <span style="color: #10b981;">${stats.episodes_completed || 0} completed</span> /
                        <span style="color: #f59e0b;">${stats.episodes_pending || 0} pending</span>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Threads</div>
                    <div class="metric-value metric-small">
                        <span style="color: #3b82f6;">${stats.threads_scheduled || 0} scheduled</span> /
                        <span style="color: #10b981;">${stats.threads_posted || 0} posted</span>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Tweets Created</div>
                    <div class="metric-value">${stats.total_tweets_created || 0}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Insights Extracted</div>
                    <div class="metric-value metric-small">${stats.insights_extracted || 0}</div>
                </div>
            `;
        }

        function updateHealthChecks(checks) {
            const icons = { healthy: '✅', warning: '⚠️', critical: '❌', unknown: '❓' };

            document.getElementById('health-checks').innerHTML = checks.map(check => `
                <div class="health-check">
                    <span class="health-icon">${icons[check.status] || '❓'}</span>
                    <div>
                        <strong>${check.name}</strong><br>
                        <span style="color: #94a3b8; font-size: 0.9em;">${check.message}</span>
                    </div>
                </div>
            `).join('');
        }

        function updateHookPerformance(hooks) {
            if (!hooks || Object.keys(hooks).length === 0) {
                document.getElementById('hook-performance').innerHTML = `
                    <p style="color: #94a3b8; text-align: center; padding: 1rem;">
                        No hook pattern data yet.<br>
                        <small>Sync Twitter metrics after posting tweets to see which hooks perform best.</small>
                    </p>
                `;
                return;
            }

            const sorted = Object.entries(hooks).sort((a, b) => b[1].avg_engagement_rate - a[1].avg_engagement_rate);

            document.getElementById('hook-performance').innerHTML = sorted.map(([pattern, data]) => `
                <div class="metric">
                    <div class="metric-label">${pattern}</div>
                    <div class="metric-value metric-small">${data.avg_engagement_rate}%</div>
                    <div style="color: #94a3b8; font-size: 0.85em;">${data.count} tweets</div>
                </div>
            `).join('');
        }

        function updateTimePerformance(timeData) {
            if (!timeData || Object.keys(timeData).length === 0) {
                document.getElementById('time-performance').innerHTML = `
                    <p style="color: #94a3b8; text-align: center; padding: 1rem;">
                        No time-of-day data yet.<br>
                        <small>Sync Twitter metrics to see which posting times get the best engagement.</small>
                    </p>
                `;
                return;
            }

            const maxRate = Math.max(...Object.values(timeData).map(d => d.avg_engagement_rate));

            document.getElementById('time-performance').innerHTML = `
                <div class="chart-container">
                    ${Object.entries(timeData).sort((a, b) => parseInt(a[0]) - parseInt(b[0])).map(([hour, data]) => {
                        const height = (data.avg_engagement_rate / maxRate * 100);
                        return `
                            <div class="chart-bar" style="height: ${height}%">
                                <span class="chart-label">${hour}h</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        }

        function updateABTests(tests) {
            if (!tests || tests.length === 0) {
                document.getElementById('ab-tests').innerHTML = '<p style="color: #94a3b8;">No active tests</p>';
                return;
            }

            document.getElementById('ab-tests').innerHTML = tests.map(test => `
                <div class="metric">
                    <div class="metric-label">${test.test_name} (${test.test_type})</div>
                    <div style="margin-top: 10px;">
                        ${test.variants.map(v => `
                            <div style="padding: 8px; background: #0f172a; margin: 5px 0; border-radius: 4px;">
                                ${v.name}: ${v.avg_engagement_rate}% (${v.tweet_count} tweets)
                            </div>
                        `).join('')}
                    </div>
                    ${test.winner_id ? `<div style="color: #10b981; margin-top: 10px;">🏆 Winner: ${test.winner_id}</div>` : ''}
                </div>
            `).join('');
        }

        function updateRecommendations(recs) {
            if (!recs || recs.length === 0) {
                document.getElementById('recommendations').innerHTML = '<p style="color: #94a3b8;">No recommendations yet - need more data</p>';
                return;
            }

            document.getElementById('recommendations').innerHTML = recs.slice(0, 5).map(rec => `
                <div class="recommendation">
                    <h3>${rec.category.toUpperCase()}: ${rec.action} ${rec.target}</h3>
                    <p style="color: #e2e8f0; margin: 8px 0;">${rec.reasoning}</p>
                    <p style="color: #94a3b8; font-size: 0.9em;">
                        Current: ${rec.current} → Recommended: ${rec.recommended}
                    </p>
                    <p style="color: #10b981; font-weight: 600;">
                        Expected improvement: +${rec.expected_improvement.toFixed(1)}%
                    </p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${rec.confidence * 100}%"></div>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.85em; margin-top: 5px;">
                        Confidence: ${(rec.confidence * 100).toFixed(0)}%
                    </p>
                </div>
            `).join('');
        }

        function updateTweetTimeline(data) {
            const timelineDiv = document.getElementById('tweet-timeline');

            if (!data.tweets || data.tweets.length === 0) {
                timelineDiv.innerHTML = '<p style="color: #94a3b8; text-align: center; padding: 2rem;">No threads scheduled yet. Run the pipeline to generate threads!</p>';
                return;
            }

            // Next item callout
            const nextItem = data.next_tweet;
            let html = '';

            if (nextItem) {
                const isThread = nextItem.item_type === 'thread' || nextItem.tweet_count > 1;
                const typeLabel = isThread ? `🧵 ${nextItem.tweet_count}-Tweet Thread` : '📝 Tweet';
                const hasThumb = nextItem.has_thumbnail ? '🖼️' : '';

                html += `
                    <div class="next-tweet-callout" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
                        <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 0.5rem;">⏰ Next: ${typeLabel} ${hasThumb}</div>
                        <div style="font-size: 2em; font-weight: 700; margin-bottom: 0.5rem;">${nextItem.time_until}</div>
                        <div style="opacity: 0.9; margin-bottom: 0.5rem;">${nextItem.scheduled_time_formatted}</div>
                        <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <div style="font-size: 0.9em; opacity: 0.8; margin-bottom: 0.3rem;">📚 ${nextItem.podcast_name} ${nextItem.podcast_handle ? '(' + nextItem.podcast_handle + ')' : ''}</div>
                            ${nextItem.episode_title ? '<div style="font-size: 0.85em; opacity: 0.7; margin-bottom: 0.3rem;">' + nextItem.episode_title + '</div>' : ''}
                            <div style="font-style: italic;">"${nextItem.preview}"</div>
                        </div>
                        <button onclick="postTweetNow('${nextItem.tweet_id}')" style="margin-top: 1rem; padding: 0.75rem 1.5rem; background: #10b981; border: none; border-radius: 8px; color: white; font-weight: 600; cursor: pointer; font-size: 1em;">
                            🚀 Post ${isThread ? 'Thread' : 'Tweet'} Now
                        </button>
                    </div>
                `;
            }

            // Timeline list - show ALL threads
            html += '<div class="timeline-list" style="max-height: 600px; overflow-y: auto;">';
            data.tweets.forEach((tweet, index) => {
                const statusColors = {
                    'overdue': '#ef4444',
                    'imminent': '#f59e0b',
                    'today': '#10b981',
                    'future': '#6366f1'
                };
                const color = statusColors[tweet.status_class] || '#6366f1';
                const isThread = tweet.item_type === 'thread' || tweet.tweet_count > 1;
                const typeIcon = isThread ? '🧵' : '📝';
                const typeLabel = isThread ? `${tweet.tweet_count} tweets` : 'single';
                const hasThumb = tweet.has_thumbnail ? ' 🖼️' : '';

                html += `
                    <div class="timeline-item" id="thread-${tweet.tweet_id}" style="border-left: 3px solid ${color}; padding-left: 1rem; margin-bottom: 1rem; background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 1rem;">
                            <div style="flex: 1; min-width: 200px;">
                                <div style="font-weight: 600; color: ${color}; margin-bottom: 0.3rem; font-size: 1.1em;">
                                    ${tweet.time_until} ${typeIcon}
                                </div>
                                <div style="font-size: 0.9em; color: #94a3b8; margin-bottom: 0.3rem;">
                                    ${tweet.scheduled_time_formatted} • ${typeLabel}${hasThumb}
                                </div>
                                <div style="font-size: 0.85em; color: #cbd5e1; margin-bottom: 0.5rem;">
                                    📚 ${tweet.podcast_name} ${tweet.podcast_handle ? '(' + tweet.podcast_handle + ')' : ''} • @${tweet.account_name}
                                </div>
                                ${tweet.episode_title ? '<div style="font-size: 0.8em; color: #a78bfa; margin-bottom: 0.3rem;">' + tweet.episode_title + '</div>' : ''}
                                <div style="font-size: 0.9em; font-style: italic; opacity: 0.8; margin-bottom: 0.75rem;">
                                    "${tweet.preview}"
                                </div>
                            </div>
                            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                                <button onclick="postTweetNow('${tweet.tweet_id}')" style="padding: 0.5rem 1rem; background: #3b82f6; border: none; border-radius: 6px; color: white; font-weight: 600; cursor: pointer; font-size: 0.85em;">
                                    📤 Post Now
                                </button>
                                <button onclick="deleteThread('${tweet.tweet_id}')" style="padding: 0.5rem 1rem; background: #ef4444; border: none; border-radius: 6px; color: white; font-weight: 600; cursor: pointer; font-size: 0.85em;">
                                    🗑️ Delete
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += '</div>';
            html += '<div style="text-align: center; margin-top: 1.5rem; color: #94a3b8; font-size: 0.9em;">Total: ' + data.total + ' threads scheduled</div>';

            timelineDiv.innerHTML = html;
        }

        async function postTweetNow(tweetId) {
            if (!confirm('Post this tweet immediately? This cannot be undone.')) return;

            const button = event.target;
            const originalText = button.textContent;
            button.disabled = true;
            button.textContent = 'Posting...';

            try {
                const result = await fetch('/api/post-tweet/' + tweetId, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                }).then(r => r.json());

                if (result.status === 'success') {
                    alert('✅ Tweet posted successfully!');
                    updateDashboard(); // Refresh timeline
                } else {
                    alert('❌ Failed to post tweet: ' + result.message);
                    button.disabled = false;
                    button.textContent = originalText;
                }
            } catch (error) {
                alert('❌ Error posting tweet: ' + error.message);
                button.disabled = false;
                button.textContent = originalText;
            }
        }

        async function deleteThread(threadId) {
            if (!confirm('Delete this thread? This cannot be undone.')) return;

            const button = event.target;
            const originalText = button.textContent;
            button.disabled = true;
            button.textContent = 'Deleting...';

            try {
                const result = await fetch('/api/delete-thread/' + threadId, {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' }
                }).then(r => r.json());

                if (result.status === 'success') {
                    // Remove the thread element from DOM immediately
                    const threadElement = document.getElementById('thread-' + threadId);
                    if (threadElement) {
                        threadElement.style.transition = 'opacity 0.3s, transform 0.3s';
                        threadElement.style.opacity = '0';
                        threadElement.style.transform = 'translateX(-20px)';
                        setTimeout(() => threadElement.remove(), 300);
                    }
                    // Refresh the full dashboard after animation
                    setTimeout(() => updateDashboard(), 400);
                } else {
                    alert('❌ Failed to delete thread: ' + result.message);
                    button.disabled = false;
                    button.textContent = originalText;
                }
            } catch (error) {
                alert('❌ Error deleting thread: ' + error.message);
                button.disabled = false;
                button.textContent = originalText;
            }
        }

        async function syncTwitterMetrics() {
            const button = event.target;
            const originalText = button.innerHTML;
            button.disabled = true;
            button.innerHTML = '⏳ Syncing...';

            try {
                const result = await fetch('/api/sync-twitter-metrics', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                }).then(r => r.json());

                if (result.status === 'success') {
                    // Calculate totals from results
                    let totalLikes = 0, totalRetweets = 0, totalImpressions = 0;
                    if (result.results) {
                        result.results.forEach(r => {
                            totalLikes += r.likes || 0;
                            totalRetweets += r.retweets || 0;
                            totalImpressions += r.impressions || 0;
                        });
                    }
                    alert('✅ Synced metrics for ' + result.synced_count + ' threads!\\n' +
                          'Total: ' + totalLikes + ' likes, ' + totalRetweets + ' retweets, ' +
                          totalImpressions + ' impressions');
                    // Refresh dashboard to show updated metrics
                    updateDashboard();
                } else {
                    alert('❌ Sync failed: ' + result.message);
                }
            } catch (error) {
                alert('❌ Error syncing metrics: ' + error.message);
            } finally {
                button.disabled = false;
                button.innerHTML = originalText;
            }
        }

        async function testApiConnection() {
            const button = event.target;
            const originalText = button.innerHTML;
            const resultDiv = document.getElementById('api-test-result');
            button.disabled = true;
            button.innerHTML = '⏳ Testing...';
            resultDiv.style.display = 'none';

            try {
                const result = await fetch('/api/test-api-connection', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                }).then(r => r.json());

                resultDiv.style.display = 'block';
                if (result.status === 'ok') {
                    const check = result.check;
                    const statusColor = check.status === 'healthy' ? '#10b981' :
                                       check.status === 'warning' ? '#f59e0b' : '#ef4444';
                    resultDiv.innerHTML = '<div style="padding: 10px; border-radius: 8px; background: ' + statusColor + '20; border: 1px solid ' + statusColor + ';">' +
                        '<strong style="color: ' + statusColor + ';">' + check.status.toUpperCase() + '</strong>: ' + check.message +
                        '</div>';
                } else {
                    resultDiv.innerHTML = '<div style="padding: 10px; border-radius: 8px; background: #ef444420; border: 1px solid #ef4444;">' +
                        '<strong style="color: #ef4444;">ERROR</strong>: ' + result.message +
                        '</div>';
                }
            } catch (error) {
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<div style="padding: 10px; border-radius: 8px; background: #ef444420; border: 1px solid #ef4444;">' +
                    '<strong style="color: #ef4444;">ERROR</strong>: ' + error.message +
                    '</div>';
            } finally {
                button.disabled = false;
                button.innerHTML = originalText;
            }
        }

        async function applyOptimizations() {
            if (!confirm('Apply high-confidence optimizations automatically?')) return;

            try {
                const result = await fetch('/api/optimizations/apply', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ confidence_threshold: 0.8, max_changes: 3 })
                }).then(r => r.json());

                alert(`✅ Applied ${result.applied_count} optimizations!`);
                updateDashboard();
            } catch (error) {
                alert('❌ Failed to apply optimizations: ' + error.message);
            }
        }

        // Tab switching
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
                tab.style.display = 'none';
            });

            // Remove active from all buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected tab
            const selectedTab = document.getElementById('tab-' + tabName);
            if (selectedTab) {
                selectedTab.classList.add('active');
                selectedTab.style.display = 'block';
            }

            // Activate button
            event.target.classList.add('active');

            // Load tab-specific data
            if (tabName === 'analytics') {
                loadAnalytics();
            } else if (tabName === 'threads') {
                loadPostedThreads();
            } else if (tabName === 'factcheck') {
                loadFactcheck();
            }
        }

        // Load Analytics Data
        async function loadAnalytics() {
            try {
                // Fetch overview
                const overview = await fetch('/api/analytics/overview').then(r => r.json());
                updateAnalyticsOverview(overview.data);

                // Fetch podcast stats
                const podcasts = await fetch('/api/analytics/by-podcast').then(r => r.json());
                updateAnalyticsPodcasts(podcasts.data);

                // Fetch timeline chart
                const timeline = await fetch('/api/analytics/timeline-chart?days=30').then(r => r.json());
                updateAnalyticsTimeline(timeline.data);

                // Fetch follower data
                const followers = await fetch('/api/analytics/followers?days=30').then(r => r.json());
                updateFollowerChart(followers);

            } catch (error) {
                console.error('Failed to load analytics:', error);
            }
        }

        // Load Fact-Check Data
        async function loadFactcheck() {
            try {
                // Fetch overview
                const overview = await fetch('/api/factcheck/overview').then(r => r.json());
                updateFactcheckOverview(overview.data);

                // Fetch debunk threads
                const threads = await fetch('/api/factcheck/debunk-threads').then(r => r.json());
                updateDebunkThreads(threads.data);

                // Fetch false/misleading claims
                const claims = await fetch('/api/factcheck/claims?verdict=false&limit=20').then(r => r.json());
                const misleading = await fetch('/api/factcheck/claims?verdict=misleading&limit=20').then(r => r.json());
                updateFactcheckClaims([...claims.data.claims, ...misleading.data.claims]);

            } catch (error) {
                console.error('Failed to load factcheck data:', error);
            }
        }

        function updateFactcheckOverview(data) {
            const container = document.getElementById('factcheck-overview');
            if (!data) {
                container.innerHTML = '<div class="loading">No data available</div>';
                return;
            }

            container.innerHTML = `
                <div class="metric">
                    <div class="metric-label">Episodes Fact-Checked</div>
                    <div class="metric-value">${data.total_runs || 0}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Claims Verified</div>
                    <div class="metric-value">${data.total_claims || 0}</div>
                </div>
                <div class="metric" style="border-left-color: #ef4444;">
                    <div class="metric-label">False Claims Found</div>
                    <div class="metric-value" style="color: #ef4444;">${data.false_claims || 0}</div>
                </div>
                <div class="metric" style="border-left-color: #f59e0b;">
                    <div class="metric-label">Misleading Claims</div>
                    <div class="metric-value" style="color: #f59e0b;">${data.misleading_claims || 0}</div>
                </div>
                <div class="metric" style="border-left-color: #10b981;">
                    <div class="metric-label">Debunk Threads Scheduled</div>
                    <div class="metric-value" style="color: #10b981;">${data.debunk_threads_scheduled || 0}</div>
                </div>
            `;

            // Update runs list
            const runsContainer = document.getElementById('factcheck-runs');
            if (data.recent_runs && data.recent_runs.length > 0) {
                runsContainer.innerHTML = data.recent_runs.map(run => `
                    <div class="health-check" style="border-left: 4px solid ${run.debunk_scheduled ? '#10b981' : '#64748b'};">
                        <div style="flex: 1;">
                            <div style="font-weight: 600;">${run.podcast_name}</div>
                            <div style="font-size: 0.85em; color: #94a3b8;">${run.episode_title?.substring(0, 50)}...</div>
                            <div style="font-size: 0.8em; color: #64748b; margin-top: 4px;">
                                ${run.total_claims} claims checked •
                                <span style="color: #ef4444;">${run.false_claims} false</span> •
                                <span style="color: #f59e0b;">${run.misleading_claims} misleading</span>
                                ${run.debunk_scheduled ? ' • <span style="color: #10b981;">Debunk scheduled</span>' : ''}
                            </div>
                        </div>
                        <div style="text-align: right; font-size: 0.75em; color: #64748b;">
                            ${new Date(run.created_at).toLocaleString()}
                        </div>
                    </div>
                `).join('');
            } else {
                runsContainer.innerHTML = '<div class="loading">No fact-check runs yet. New episodes will be fact-checked automatically.</div>';
            }
        }

        function updateDebunkThreads(data) {
            const container = document.getElementById('debunk-threads');
            if (!data || !data.threads || data.threads.length === 0) {
                container.innerHTML = '<div class="loading">No PodDebunker threads yet</div>';
                return;
            }

            container.innerHTML = data.threads.map(thread => {
                const statusColor = thread.status === 'posted' ? '#10b981' :
                                   thread.status === 'scheduled' ? '#3b82f6' : '#f59e0b';
                return `
                    <div class="health-check" style="border-left: 4px solid ${statusColor};">
                        <div style="flex: 1;">
                            <div style="font-weight: 600;">${thread.podcast_name}</div>
                            <div style="font-size: 0.85em; color: #94a3b8;">${thread.episode_title?.substring(0, 50)}...</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="status-badge" style="background: ${statusColor}; font-size: 0.7em;">${thread.status}</span>
                            <div style="font-size: 0.75em; color: #64748b; margin-top: 4px;">
                                ${thread.scheduled_time ? new Date(thread.scheduled_time).toLocaleString() : ''}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function updateFactcheckClaims(claims) {
            const container = document.getElementById('factcheck-claims');
            if (!claims || claims.length === 0) {
                container.innerHTML = '<div class="loading">No false/misleading claims found yet</div>';
                return;
            }

            container.innerHTML = claims.slice(0, 10).map(claim => {
                const verdictColor = claim.verdict === 'false' ? '#ef4444' : '#f59e0b';
                return `
                    <div class="health-check" style="border-left: 4px solid ${verdictColor}; margin-bottom: 8px;">
                        <div style="flex: 1;">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                <span class="status-badge" style="background: ${verdictColor}; font-size: 0.7em;">${claim.verdict}</span>
                                <span style="font-size: 0.8em; color: #64748b;">${claim.podcast_name}</span>
                            </div>
                            <div style="font-weight: 500;">"${claim.claim_text?.substring(0, 150)}..."</div>
                            ${claim.correction ? `<div style="font-size: 0.85em; color: #10b981; margin-top: 4px;">✓ ${claim.correction.substring(0, 100)}...</div>` : ''}
                            ${claim.source_name ? `<div style="font-size: 0.75em; color: #64748b; margin-top: 4px;">Source: ${claim.source_name}</div>` : ''}
                        </div>
                        <div style="text-align: right; font-size: 0.75em; color: #64748b;">
                            ${(claim.confidence * 100).toFixed(0)}% confidence
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Record follower count
        async function recordFollowerCount() {
            const input = document.getElementById('follower-count-input');
            const count = parseInt(input.value);

            if (!count || count < 0) {
                alert('Please enter a valid follower count');
                return;
            }

            try {
                const result = await fetch('/api/analytics/followers/record', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        account_name: 'podcasts_tldr',
                        follower_count: count,
                        source: 'manual'
                    })
                }).then(r => r.json());

                if (result.status === 'success') {
                    alert('Follower count recorded successfully!');
                    input.value = '';
                    // Refresh follower data
                    const followers = await fetch('/api/analytics/followers?days=30').then(r => r.json());
                    updateFollowerChart(followers);
                } else {
                    alert('Failed to record: ' + result.message);
                }
            } catch (error) {
                alert('Error recording follower count: ' + error.message);
            }
        }

        function updateFollowerChart(data) {
            // Update current count
            const currentDiv = document.getElementById('follower-current');
            if (data.current && Object.keys(data.current).length > 0) {
                const account = Object.keys(data.current)[0];
                const info = data.current[account];
                currentDiv.innerHTML = `
                    <div class="metric">
                        <div class="metric-label">Current Followers</div>
                        <div class="metric-value">${info.count.toLocaleString()}</div>
                        <div style="font-size: 0.8em; color: #94a3b8; margin-top: 0.5rem;">
                            As of ${new Date(info.as_of).toLocaleDateString()}
                        </div>
                    </div>
                `;
            } else {
                currentDiv.innerHTML = `
                    <div class="metric">
                        <p style="color: #94a3b8; font-size: 0.9em;">
                            No follower data yet.<br>
                            Enter your current follower count above to start tracking growth.
                        </p>
                    </div>
                `;
            }

            // Update chart
            const chartDiv = document.getElementById('follower-chart');
            if (!data.data || data.data.length === 0) {
                chartDiv.innerHTML = `
                    <div style="background: #0f172a; padding: 2rem; border-radius: 8px; text-align: center;">
                        <p style="color: #94a3b8;">
                            No follower history yet.<br>
                            Record your follower count daily to track growth over time.
                        </p>
                    </div>
                `;
                return;
            }

            const maxCount = Math.max(...data.data.map(d => d.count));
            const minCount = Math.min(...data.data.map(d => d.count));
            const range = maxCount - minCount || 1;

            let html = '<div class="timeline-bar-chart" style="height: 150px;">';
            data.data.forEach(point => {
                const height = ((point.count - minCount) / range * 80) + 20; // 20-100% range
                const date = new Date(point.date);
                const dayLabel = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

                html += `
                    <div class="timeline-bar" style="height: ${height}%; background: linear-gradient(180deg, #10b981 0%, #065f46 100%);">
                        <div class="tooltip">
                            <strong>${dayLabel}</strong><br>
                            ${point.count.toLocaleString()} followers
                        </div>
                    </div>
                `;
            });
            html += '</div>';

            // Growth stats
            if (data.data.length >= 2) {
                const firstCount = data.data[0].count;
                const lastCount = data.data[data.data.length - 1].count;
                const growth = lastCount - firstCount;
                const growthPercent = ((growth / firstCount) * 100).toFixed(1);
                const growthColor = growth >= 0 ? '#10b981' : '#ef4444';
                const growthIcon = growth >= 0 ? '📈' : '📉';

                html += `
                    <div style="margin-top: 1rem; padding: 1rem; background: #0f172a; border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #94a3b8;">Growth (${data.period_days} days)</span>
                            <span style="color: ${growthColor}; font-weight: 600; font-size: 1.2em;">
                                ${growthIcon} ${growth >= 0 ? '+' : ''}${growth.toLocaleString()} (${growthPercent}%)
                            </span>
                        </div>
                    </div>
                `;
            }

            chartDiv.innerHTML = html;
        }

        function updateAnalyticsOverview(data) {
            document.getElementById('analytics-overview').innerHTML = `
                <div class="grid" style="grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div class="big-stat">
                        <div class="value">${data.threads.posted}</div>
                        <div class="label">Threads Posted</div>
                    </div>
                    <div class="big-stat">
                        <div class="value">${data.tweets_posted}</div>
                        <div class="label">Total Tweets</div>
                    </div>
                </div>
                <div class="metric" style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Scheduled:</span>
                        <span style="color: #3b82f6; font-weight: 600;">${data.threads.scheduled}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                        <span>Failed:</span>
                        <span style="color: #ef4444; font-weight: 600;">${data.threads.failed}</span>
                    </div>
                </div>
            `;

            document.getElementById('analytics-period').innerHTML = `
                <div class="metric">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Today:</span>
                        <span style="color: #10b981; font-weight: 600;">${data.posts_today} threads</span>
                    </div>
                </div>
                <div class="metric">
                    <div style="display: flex; justify-content: space-between;">
                        <span>This Week:</span>
                        <span style="color: #3b82f6; font-weight: 600;">${data.posts_this_week} threads</span>
                    </div>
                </div>
                <div class="metric">
                    <div style="display: flex; justify-content: space-between;">
                        <span>This Month:</span>
                        <span style="color: #a78bfa; font-weight: 600;">${data.posts_this_month} threads</span>
                    </div>
                </div>
                ${data.first_post_date ? `
                <div style="margin-top: 1rem; font-size: 0.85em; color: #94a3b8;">
                    First post: ${new Date(data.first_post_date).toLocaleDateString()}<br>
                    Last post: ${new Date(data.last_post_date).toLocaleDateString()}
                </div>
                ` : ''}
            `;
        }

        function updateAnalyticsPodcasts(podcasts) {
            if (!podcasts || podcasts.length === 0) {
                document.getElementById('analytics-podcasts').innerHTML = '<p style="color: #94a3b8;">No podcasts yet</p>';
                return;
            }

            document.getElementById('analytics-podcasts').innerHTML = podcasts.slice(0, 10).map(p => `
                <div class="metric" style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>${p.podcast_name}</strong>
                            ${p.podcast_handle ? '<span style="color: #94a3b8; font-size: 0.85em;"> ' + p.podcast_handle + '</span>' : ''}
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #10b981;">${p.posted}</span>
                            <span style="color: #94a3b8; font-size: 0.85em;">posted</span>
                            ${p.scheduled > 0 ? '<span style="color: #3b82f6; margin-left: 0.5rem;">' + p.scheduled + '</span><span style="color: #94a3b8; font-size: 0.85em;"> scheduled</span>' : ''}
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function updateAnalyticsTimeline(data) {
            if (!data || data.length === 0) {
                document.getElementById('analytics-timeline-chart').innerHTML = `
                    <p style="color: #94a3b8; text-align: center; padding: 2rem;">
                        No posting data yet. Timeline will appear after threads are posted.
                    </p>
                `;
                return;
            }

            const maxThreads = Math.max(...data.map(d => d.threads));

            let html = '<div class="timeline-bar-chart">';
            data.forEach(day => {
                const height = maxThreads > 0 ? (day.threads / maxThreads * 100) : 0;
                const date = new Date(day.date);
                const dayLabel = date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

                html += `
                    <div class="timeline-bar" style="height: ${Math.max(height, 5)}%;">
                        <div class="tooltip">
                            <strong>${dayLabel}</strong><br>
                            ${day.threads} threads<br>
                            ${day.tweets} tweets
                        </div>
                    </div>
                `;
            });
            html += '</div>';

            html += `
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8em; color: #94a3b8;">
                    <span>${data.length > 0 ? new Date(data[0].date).toLocaleDateString() : ''}</span>
                    <span>${data.length > 0 ? new Date(data[data.length - 1].date).toLocaleDateString() : ''}</span>
                </div>
            `;

            document.getElementById('analytics-timeline-chart').innerHTML = html;
        }

        // Load Posted Threads
        async function loadPostedThreads() {
            try {
                const result = await fetch('/api/analytics/posted-threads').then(r => r.json());
                updatePostedThreadsList(result.data);
            } catch (error) {
                console.error('Failed to load posted threads:', error);
            }
        }

        function updatePostedThreadsList(threads) {
            if (!threads || threads.length === 0) {
                document.getElementById('posted-threads-list').innerHTML = `
                    <p style="color: #94a3b8; text-align: center; padding: 2rem;">
                        No threads posted yet. Run the pipeline to generate and post threads!
                    </p>
                `;
                return;
            }

            let html = `<div style="margin-bottom: 1rem; color: #94a3b8;">Total: ${threads.length} threads posted</div>`;
            html += '<div style="max-height: 600px; overflow-y: auto;">';

            threads.forEach(thread => {
                const postedDate = thread.posted_time ? new Date(thread.posted_time).toLocaleString() : 'Unknown';
                const tweetIds = thread.all_tweet_ids || [];
                const hasTweetIds = tweetIds.length > 0;

                html += `
                    <div class="thread-item">
                        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 1rem;">
                            <div style="flex: 1; min-width: 200px;">
                                <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem;">
                                    ${thread.podcast_name || 'Unknown Podcast'}
                                    ${thread.podcast_handle ? '<span style="color: #94a3b8; font-size: 0.85em;">' + thread.podcast_handle + '</span>' : ''}
                                </div>
                                <div style="font-size: 0.9em; color: #a78bfa; margin-bottom: 0.5rem;">
                                    ${thread.episode_title || 'Untitled Episode'}
                                </div>
                                <div style="font-size: 0.8em; color: #94a3b8;">
                                    Posted: ${postedDate} • ${thread.tweet_count} tweets
                                </div>
                            </div>
                            <div style="text-align: right;">
                                ${thread.thread_url ? `
                                    <a href="${thread.thread_url}" target="_blank" style="display: inline-block; padding: 0.5rem 1rem; background: #3b82f6; border-radius: 6px; color: white; font-size: 0.85em; font-weight: 600;">
                                        View on X.com
                                    </a>
                                ` : '<span style="color: #94a3b8; font-size: 0.85em;">No tweet ID</span>'}
                            </div>
                        </div>
                        ${hasTweetIds ? `
                            <details style="margin-top: 0.75rem;">
                                <summary style="cursor: pointer; color: #94a3b8; font-size: 0.85em;">
                                    Tweet IDs (${tweetIds.length})
                                </summary>
                                <div style="margin-top: 0.5rem; padding: 0.5rem; background: rgba(0,0,0,0.3); border-radius: 4px; font-family: monospace; font-size: 0.75em; color: #94a3b8;">
                                    ${tweetIds.map((id, i) => `<a href="https://x.com/i/status/${id}" target="_blank" style="color: #667eea;">${i + 1}. ${id}</a>`).join('<br>')}
                                </div>
                            </details>
                        ` : ''}
                    </div>
                `;
            });

            html += '</div>';
            document.getElementById('posted-threads-list').innerHTML = html;
        }

        // Initial load and auto-refresh every 30 seconds
        updateDashboard();
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>'''

    with open(template_path, 'w') as f:
        f.write(html_content)

    logger.info(f"✅ Dashboard template created: {template_path}")


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run the web dashboard."""

    # Initialize components
    init_components()

    # Create template
    create_dashboard_template()

    # Run Flask app
    logger.info(f"🌐 Starting dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Web Dashboard")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_dashboard(host=args.host, port=args.port, debug=args.debug)
