# Deployment Options

This directory contains deployment configurations for running Podcasts TLDR as a 24/7 autonomous service.

## Available Deployment Methods

### 1. Docker (Recommended)

**Best for**: Most use cases, easy setup, portability

See root `docker-compose.yml` and `Dockerfile`

```bash
cd /path/to/AI-tweets
docker-compose up -d
```

**Pros**:
- ✅ Easiest to set up
- ✅ Isolated environment
- ✅ Portable across systems
- ✅ Resource limits built-in
- ✅ Easy updates

**Cons**:
- ❌ Requires Docker installed
- ❌ Slight performance overhead

---

### 2. Systemd Service (Linux)

**Best for**: Dedicated Linux servers, production deployments

See `systemd/` directory

```bash
cd /path/to/AI-tweets
sudo ./deployment/systemd/install.sh
```

**Pros**:
- ✅ Native system integration
- ✅ Automatic start on boot
- ✅ System logging (journald)
- ✅ Best performance
- ✅ System-level resource management

**Cons**:
- ❌ Linux only
- ❌ Requires root access
- ❌ More setup steps

---

### 3. Manual Python Process

**Best for**: Development, testing, non-Linux systems

```bash
cd /path/to/AI-tweets
python src/autonomous_orchestrator.py
```

**Pros**:
- ✅ Works on any OS
- ✅ No special setup
- ✅ Easy debugging

**Cons**:
- ❌ Not persistent (stops when terminal closes)
- ❌ No auto-restart on failure
- ❌ Must run manually

**To run persistently**:
```bash
# Using nohup (Linux/Mac)
nohup python src/autonomous_orchestrator.py > orchestrator.log 2>&1 &

# Using screen (Linux/Mac)
screen -S podcasts
python src/autonomous_orchestrator.py
# Ctrl+A, D to detach

# Using tmux (Linux/Mac)
tmux new -s podcasts
python src/autonomous_orchestrator.py
# Ctrl+B, D to detach
```

---

## Comparison Matrix

| Feature | Docker | Systemd | Manual |
|---------|--------|---------|--------|
| Setup Difficulty | ⭐⭐ Easy | ⭐⭐⭐ Medium | ⭐ Very Easy |
| Auto-restart | ✅ Yes | ✅ Yes | ❌ No |
| Resource Limits | ✅ Built-in | ✅ Built-in | ❌ Manual |
| Isolation | ✅ Yes | ⚠️ Partial | ❌ No |
| Cross-platform | ✅ Yes | ❌ Linux only | ✅ Yes |
| Performance | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Best |
| Production-ready | ✅ Yes | ✅ Yes | ❌ No |

---

## Quick Decision Guide

**Choose Docker if**:
- You want the easiest setup
- You might move to different servers
- You want containerization
- You're on Linux, Mac, or Windows

**Choose Systemd if**:
- You have a dedicated Linux server
- You want native system integration
- You need maximum performance
- You want system-level logging

**Choose Manual if**:
- You're just testing
- You're developing new features
- You don't need 24/7 operation

---

## Post-Deployment

After deploying with any method, see [AUTONOMOUS_DEPLOYMENT.md](../AUTONOMOUS_DEPLOYMENT.md) for:
- Configuration tuning
- Monitoring and alerting
- Health checks
- Troubleshooting

---

## Support

For issues with deployment:
1. Check logs (see deployment method above)
2. Review [AUTONOMOUS_DEPLOYMENT.md](../AUTONOMOUS_DEPLOYMENT.md)
3. Open an issue with logs and config (remove API keys!)
