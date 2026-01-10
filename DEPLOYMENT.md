# Windows GPU Deployment Guide

## Current Status (2025-12-15)

GPU Docker deployment files are ready. The pipeline should run on Windows with NVIDIA GPU acceleration.

## Files for GPU Deployment

| File | Purpose |
|------|---------|
| `Dockerfile.gpu` | NVIDIA CUDA 12.1 base image with Python 3.11 |
| `docker-compose.gpu.yml` | GPU runtime with NVIDIA device reservation |
| `requirements-gpu.txt` | PyTorch with CUDA 12.1 support |
| `viral_config.example.json` | Template config (copy to viral_config.json) |

## Setup Steps

### 1. Create config file
```powershell
copy viral_config.example.json viral_config.json
```

### 2. Edit viral_config.json
Add your X/Twitter API credentials in the `x_accounts` section:
```json
"x_accounts": {
  "podcasts_tldr": {
    "consumer_key": "YOUR_MAIN_API_KEY",
    "consumer_secret": "YOUR_MAIN_API_SECRET",
    "access_token": "YOUR_MAIN_ACCESS_TOKEN",
    "access_token_secret": "YOUR_MAIN_ACCESS_TOKEN_SECRET",
    "bearer_token": "YOUR_MAIN_BEARER_TOKEN"
  }
}
```

### 3. Create .env file
```powershell
copy .env.example .env
```
Add your API keys:
```
OPENROUTER_API_KEY=your_key_here
USE_OPENROUTER=true
```

### 4. Build and run
```powershell
docker-compose -f docker-compose.gpu.yml up --build -d
```

### 5. Verify GPU detection
```powershell
docker exec podcasts-tldr-gpu python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 6. Check logs
```powershell
docker logs podcasts-tldr-gpu --tail 100
```

## Troubleshooting

### Container keeps restarting
```powershell
docker logs podcasts-tldr-gpu --tail 100
```
Look for Python errors - usually missing modules or config issues.

### "ModuleNotFoundError"
Some source files may not be committed. On Mac, run:
```bash
git status src/
git add src/
git commit -m "Add missing modules"
git push
```
Then pull on Windows and rebuild.

### "Is a directory: viral_config.json"
Docker created a directory instead of mounting a file:
```powershell
rmdir viral_config.json
copy viral_config.example.json viral_config.json
```

### CUDA not detected
Verify GPU passthrough works:
```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

## Architecture

```
Windows Docker Desktop (WSL2)
    └── podcasts-tldr-gpu container
        ├── NVIDIA CUDA 12.1 runtime
        ├── Python 3.11 + PyTorch (CUDA)
        ├── Whisper model (base, ~1GB VRAM)
        └── Autonomous Orchestrator
            ├── Discovery loop (every 4 hours)
            ├── Processing loop (every 30 min)
            └── Posting loop (every 5 min)
```

## Expected Performance

| Hardware | Transcription Speed |
|----------|-------------------|
| CPU only | ~30 min per 1-hour podcast |
| NVIDIA GPU | ~5-10 min per 1-hour podcast |

## Web Dashboard

Access at http://localhost:5000 after container starts.

## For Claude Code on Windows

If you're Claude Code running on Windows and encounter errors:

1. Check `docker logs podcasts-tldr-gpu` for the actual error
2. Most issues are missing files - check if the file exists on Mac but not in git
3. Config issues usually mean viral_config.json is missing or malformed
4. GPU issues mean Docker GPU passthrough isn't working - test with nvidia-smi first
