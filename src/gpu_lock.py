#!/usr/bin/env python3
"""
GPU Lock - File-based lock to serialize GPU-intensive operations.

Prevents Hebrew and English orchestrators from running transcription
simultaneously, which could cause OOM or GPU conflicts.
"""

import fcntl
import time
from pathlib import Path
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

LOCK_FILE = Path("/tmp/podcasts_gpu.lock")


@contextmanager
def gpu_lock(timeout: int = 3600, name: str = "unknown"):
    """
    Acquire GPU lock before transcription.

    Uses file-based locking (fcntl.flock) to ensure only one
    transcription process runs at a time across all orchestrators.

    Args:
        timeout: Max seconds to wait for lock (default: 1 hour)
        name: Pipeline name for logging (e.g., "Hebrew", "English")

    Raises:
        TimeoutError: If lock cannot be acquired within timeout

    Example:
        with gpu_lock(name="Hebrew"):
            transcription = transcriber.transcribe(...)
    """
    LOCK_FILE.parent.mkdir(exist_ok=True)

    # Open lock file (create if doesn't exist)
    lock_fd = open(LOCK_FILE, 'w')

    start = time.time()
    acquired = False

    try:
        while True:
            try:
                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                logger.info(f"[{name}] Acquired GPU lock")
                break
            except BlockingIOError:
                # Lock is held by another process
                elapsed = time.time() - start
                if elapsed > timeout:
                    lock_fd.close()
                    raise TimeoutError(
                        f"[{name}] GPU lock timeout after {timeout}s - "
                        f"another transcription may be stuck"
                    )
                # Log every 30 seconds
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    logger.info(
                        f"[{name}] Waiting for GPU lock ({elapsed:.0f}s elapsed)..."
                    )
                time.sleep(10)

        # Lock acquired, yield to caller
        yield

    finally:
        if acquired:
            # Release lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            logger.info(f"[{name}] Released GPU lock")
        lock_fd.close()


def is_gpu_locked() -> bool:
    """
    Check if GPU lock is currently held (non-blocking check).

    Returns:
        True if lock is held by another process, False if available
    """
    if not LOCK_FILE.exists():
        return False

    try:
        lock_fd = open(LOCK_FILE, 'w')
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            return False  # Lock was available
        except BlockingIOError:
            return True  # Lock is held
        finally:
            lock_fd.close()
    except Exception:
        return False


def force_release_lock():
    """
    Force release the GPU lock (use only for recovery).

    Warning: This should only be used if a process crashed
    while holding the lock and the lock file is stale.
    """
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
            logger.warning("Force released GPU lock")
        except Exception as e:
            logger.error(f"Could not force release lock: {e}")
