"""
Web Interface for the Podcasts TLDR Viral Content Machine.
Provides a dashboard for managing episodes, insights, tweets, and scheduling.
"""

import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from .podcast_ingestor import PodcastIngestor, EpisodeDatabase
from .viral_scheduler import ViralScheduler, TweetQueue, TweetStatus
from .viral_insight_extractor import InsightDatabase

logger = logging.getLogger(__name__)


def create_app(config_path: str = "config.json") -> Flask:
    """Create and configure Flask app."""
    
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    
    # Initialize components
    episode_db = EpisodeDatabase()
    insight_db = InsightDatabase()
    tweet_queue = TweetQueue()
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        config = {}
    
    @app.route('/')
    def dashboard():
        """Main dashboard view."""
        
        # Get recent statistics
        stats = {
            'episodes': {
                'pending': len(episode_db.get_pending_episodes(50)),
                'total': 0  # Would query total count
            },
            'insights': {
                'high_viral': len(insight_db.get_top_insights(limit=20, min_score=0.7)),
                'total': 0  # Would query total count
            },
            'tweets': tweet_queue.get_queue_stats(),
            'performance': {
                'avg_engagement': 0.0,
                'viral_tweets': 0,
                'best_time': "12:30 PM"
            }
        }
        
        # Get recent activity
        recent_episodes = episode_db.get_pending_episodes(5)
        top_insights = insight_db.get_top_insights(limit=10)
        
        return render_template('dashboard.html', 
                             stats=stats,
                             recent_episodes=recent_episodes,
                             top_insights=top_insights)
    
    @app.route('/episodes')
    def episodes_view():
        """Episodes management view."""
        
        status_filter = request.args.get('status', 'all')
        
        # Get episodes based on filter
        if status_filter == 'pending':
            episodes = episode_db.get_pending_episodes(50)
        else:
            episodes = episode_db.get_pending_episodes(100)  # All recent
        
        return render_template('episodes.html', 
                             episodes=episodes,
                             status_filter=status_filter)
    
    @app.route('/insights')
    def insights_view():
        """Insights management view."""
        
        min_score = float(request.args.get('min_score', 0.5))
        limit = int(request.args.get('limit', 20))
        
        insights = insight_db.get_top_insights(limit=limit, min_score=min_score)
        
        return render_template('insights.html',
                             insights=insights,
                             min_score=min_score,
                             limit=limit)
    
    @app.route('/tweets')
    def tweets_view():
        """Tweet queue management view."""
        
        account_filter = request.args.get('account', 'all')
        status_filter = request.args.get('status', 'all')
        
        # Get tweets from queue (simplified - would implement full filtering)
        stats = tweet_queue.get_queue_stats()
        
        return render_template('tweets.html',
                             stats=stats,
                             account_filter=account_filter,
                             status_filter=status_filter)
    
    @app.route('/api/process_episode', methods=['POST'])
    def api_process_episode():
        """API endpoint to process a specific episode."""
        
        try:
            data = request.json
            episode_id = data.get('episode_id')
            
            if not episode_id:
                return jsonify({'error': 'Episode ID required'}), 400
            
            # Mark episode as processing
            episode_db.update_episode_status(episode_id, "processing")
            
            # In a real implementation, this would trigger the full pipeline
            # For now, just simulate success
            
            return jsonify({
                'success': True,
                'message': f'Episode {episode_id} queued for processing'
            })
            
        except Exception as e:
            logger.error(f"Error processing episode: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/schedule_tweet', methods=['POST'])
    def api_schedule_tweet():
        """API endpoint to schedule or reschedule a tweet."""
        
        try:
            data = request.json
            tweet_id = data.get('tweet_id')
            new_time = data.get('scheduled_time')
            
            if not tweet_id or not new_time:
                return jsonify({'error': 'Tweet ID and scheduled time required'}), 400
            
            # Update tweet schedule
            tweet_queue.update_tweet_status(
                tweet_id, 
                TweetStatus.SCHEDULED,
                scheduled_time=new_time
            )
            
            return jsonify({
                'success': True,
                'message': f'Tweet {tweet_id} rescheduled'
            })
            
        except Exception as e:
            logger.error(f"Error scheduling tweet: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/cancel_tweet', methods=['POST'])
    def api_cancel_tweet():
        """API endpoint to cancel a scheduled tweet."""
        
        try:
            data = request.json
            tweet_id = data.get('tweet_id')
            
            if not tweet_id:
                return jsonify({'error': 'Tweet ID required'}), 400
            
            tweet_queue.update_tweet_status(tweet_id, TweetStatus.CANCELLED)
            
            return jsonify({
                'success': True,
                'message': f'Tweet {tweet_id} cancelled'
            })
            
        except Exception as e:
            logger.error(f"Error cancelling tweet: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/stats')
    def api_stats():
        """API endpoint for dashboard statistics."""
        
        try:
            stats = {
                'episodes': {
                    'pending': len(episode_db.get_pending_episodes(100)),
                    'processing': 0,  # Would query database
                    'completed': 0
                },
                'tweets': tweet_queue.get_queue_stats(),
                'insights': {
                    'total': len(insight_db.get_top_insights(1000, 0.0)),
                    'high_viral': len(insight_db.get_top_insights(100, 0.7))
                },
                'performance': {
                    'total_engagement': 0,
                    'avg_viral_score': 0.0,
                    'best_performing_format': 'thread'
                }
            }
            
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app


def create_templates():
    """Create basic HTML templates for the web interface."""
    
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Base template
    base_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Podcasts TLDR - Viral Content Machine{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
</head>
<body class="bg-gray-100">
    <nav class="bg-blue-600 text-white p-4">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-xl font-bold">🎙️ Podcasts TLDR</h1>
            <div class="space-x-4">
                <a href="/" class="hover:underline">Dashboard</a>
                <a href="/episodes" class="hover:underline">Episodes</a>
                <a href="/insights" class="hover:underline">Insights</a>
                <a href="/tweets" class="hover:underline">Tweets</a>
            </div>
        </div>
    </nav>
    
    <main class="container mx-auto p-4">
        {% block content %}{% endblock %}
    </main>
    
    <script>
        // Auto-refresh stats every 30 seconds
        setInterval(() => {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    // Update dashboard stats if on dashboard page
                    if (window.location.pathname === '/') {
                        updateDashboardStats(data);
                    }
                })
                .catch(console.error);
        }, 30000);
    </script>
</body>
</html>
'''
    
    # Dashboard template
    dashboard_template = '''
{% extends "base.html" %}

{% block content %}
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
    <!-- Stats Cards -->
    <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold text-gray-700">Episodes</h3>
        <p class="text-3xl font-bold text-blue-600">{{ stats.episodes.pending }}</p>
        <p class="text-sm text-gray-500">Pending Processing</p>
    </div>
    
    <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold text-gray-700">Insights</h3>
        <p class="text-3xl font-bold text-green-600">{{ stats.insights.high_viral }}</p>
        <p class="text-sm text-gray-500">High Viral Score</p>
    </div>
    
    <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold text-gray-700">Tweets</h3>
        <p class="text-3xl font-bold text-purple-600">{{ stats.tweets.scheduled }}</p>
        <p class="text-sm text-gray-500">Scheduled</p>
    </div>
    
    <div class="bg-white rounded-lg shadow p-6">
        <h3 class="text-lg font-semibold text-gray-700">Posted</h3>
        <p class="text-3xl font-bold text-orange-600">{{ stats.tweets.posted }}</p>
        <p class="text-sm text-gray-500">Total Posted</p>
    </div>
</div>

<div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
    <!-- Recent Episodes -->
    <div class="bg-white rounded-lg shadow">
        <div class="p-6 border-b border-gray-200">
            <h2 class="text-xl font-semibold">Recent Episodes</h2>
        </div>
        <div class="p-6">
            {% for episode in recent_episodes %}
            <div class="mb-4 p-4 border border-gray-200 rounded">
                <h3 class="font-semibold">{{ episode.podcast_name }}</h3>
                <p class="text-gray-600">{{ episode.title }}</p>
                <p class="text-sm text-gray-500">Viral Score: {{ "%.2f"|format(episode.viral_score or 0) }}</p>
                <button onclick="processEpisode('{{ episode.episode_id }}')" 
                        class="mt-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                    Process Episode
                </button>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <!-- Top Insights -->
    <div class="bg-white rounded-lg shadow">
        <div class="p-6 border-b border-gray-200">
            <h2 class="text-xl font-semibold">Top Insights</h2>
        </div>
        <div class="p-6">
            {% for insight in top_insights %}
            <div class="mb-4 p-4 border border-gray-200 rounded">
                <p class="font-medium">{{ insight.text[:100] }}...</p>
                <p class="text-sm text-gray-500">Type: {{ insight.insight_type }}</p>
                <p class="text-sm text-gray-500">Viral Score: {{ "%.2f"|format(insight.viral_score) }}</p>
            </div>
            {% endfor %}
        </div>
    </div>
</div>

<script>
function processEpisode(episodeId) {
    fetch('/api/process_episode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({episode_id: episodeId})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Episode queued for processing!');
            location.reload();
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error processing episode');
    });
}
</script>
{% endblock %}
'''
    
    # Write templates
    with open(templates_dir / "base.html", "w") as f:
        f.write(base_template)
    
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_template)
    
    logger.info("Created web interface templates")


if __name__ == "__main__":
    # Create templates
    create_templates()
    
    # Create and run app
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)