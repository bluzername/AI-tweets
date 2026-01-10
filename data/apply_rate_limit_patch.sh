#!/bin/bash
# Apply rate limit manager patch on container start
cp /app/data/x_rate_manager.py /app/src/x_rate_manager.py 2>/dev/null || true
cp /app/data/viral_scheduler_patched.py /app/src/viral_scheduler.py 2>/dev/null || true
echo "Rate limit patch applied"
