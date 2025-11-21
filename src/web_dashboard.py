#!/usr/bin/env python3
"""
Web Dashboard - Comprehensive monitoring and control interface.
Real-time view of pipeline status, performance, and manual controls.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_monitor import HealthMonitor
from src.performance_tracker import PerformanceTracker
from src.ab_testing_framework import ABTestingFramework
from src.feedback_loop_optimizer import FeedbackLoopOptimizer

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')


# Global instances
health_monitor = None
performance_tracker = None
ab_testing = None
optimizer = None


def init_components():
    """Initialize dashboard components."""
    global health_monitor, performance_tracker, ab_testing, optimizer

    try:
        health_monitor = HealthMonitor()
        performance_tracker = PerformanceTracker()
        ab_testing = ABTestingFramework()
        optimizer = FeedbackLoopOptimizer(performance_tracker, ab_testing)
        logger.info("✅ Dashboard components initialized")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")


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

        # Orchestrator metrics
        metrics_file = Path("data/orchestrator_metrics.json")
        orchestrator_metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                orchestrator_metrics = json.load(f)

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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎙️ Podcasts TLDR Dashboard</h1>
            <p class="subtitle">Autonomous Content Machine - Real-time Monitoring & Control</p>
        </header>

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

        <div class="card">
            <h2>💡 Optimization Recommendations</h2>
            <div id="recommendations">
                <div class="loading">Loading...</div>
            </div>
            <button onclick="applyOptimizations()" style="margin-top: 15px;">
                🚀 Auto-Apply High-Confidence Optimizations
            </button>
        </div>
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
                    <div class="metric-label">Avg Viral Score</div>
                    <div class="metric-value metric-small">${perf.avg_viral_score || 0}</div>
                </div>
            `;
        }

        function updateOrchestratorStats(stats) {
            document.getElementById('orchestrator-stats').innerHTML = `
                <div class="metric">
                    <div class="metric-label">Episodes Processed</div>
                    <div class="metric-value">${stats.total_processed || 0}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Tweets Created</div>
                    <div class="metric-value">${stats.total_tweets_created || 0}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">API Cost Today</div>
                    <div class="metric-value metric-small">$${(stats.api_cost_today || 0).toFixed(2)}</div>
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
                document.getElementById('hook-performance').innerHTML = '<p style="color: #94a3b8;">No data yet</p>';
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
                document.getElementById('time-performance').innerHTML = '<p style="color: #94a3b8;">No data yet</p>';
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
