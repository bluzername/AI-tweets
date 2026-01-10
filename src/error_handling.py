#!/usr/bin/env python3
"""
Robust Error Handling for Podcasts TLDR Pipeline

Provides:
- Retry decorators with exponential backoff
- Error classification (transient vs permanent)
- Circuit breaker for failing services
- Safe execution wrappers
- Error aggregation and reporting
"""

import logging
import time
import functools
import traceback
from typing import Callable, TypeVar, Optional, List, Dict, Any, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for different handling strategies."""
    TRANSIENT = "transient"      # Network timeout, rate limit - retry
    PERMANENT = "permanent"       # Invalid data, auth error - skip
    RESOURCE = "resource"         # Out of memory, disk full - pause
    UNKNOWN = "unknown"           # Unclassified - log and continue


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_type: str
    message: str
    category: ErrorCategory
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None


class ErrorClassifier:
    """Classifies errors into categories for appropriate handling."""
    
    # Transient errors - worth retrying
    TRANSIENT_PATTERNS = [
        "timeout", "timed out", "connection reset", "connection refused",
        "rate limit", "too many requests", "429", "503", "502", "504",
        "temporary", "retry", "overloaded", "capacity", "busy",
        "network", "ssl", "certificate", "dns", "socket"
    ]
    
    # Permanent errors - skip and move on
    PERMANENT_PATTERNS = [
        "not found", "404", "invalid", "malformed", "unauthorized", "401",
        "forbidden", "403", "no transcript", "disabled", "private",
        "not available", "does not exist", "authentication", "api key"
    ]
    
    # Resource errors - may need to pause/throttle
    RESOURCE_PATTERNS = [
        "memory", "disk", "space", "quota", "limit exceeded", "oom",
        "killed", "signal 9", "cannot allocate"
    ]
    
    @classmethod
    def classify(cls, error: Exception) -> ErrorCategory:
        """Classify an error into a category."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        combined = f"{error_type} {error_str}"
        
        # Check patterns
        for pattern in cls.TRANSIENT_PATTERNS:
            if pattern in combined:
                return ErrorCategory.TRANSIENT
        
        for pattern in cls.PERMANENT_PATTERNS:
            if pattern in combined:
                return ErrorCategory.PERMANENT
        
        for pattern in cls.RESOURCE_PATTERNS:
            if pattern in combined:
                return ErrorCategory.RESOURCE
        
        return ErrorCategory.UNKNOWN


class CircuitBreaker:
    """
    Circuit breaker to prevent repeated calls to failing services.
    
    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Service failing, calls fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 name: str = "default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if we can execute a call."""
        with self._lock:
            if self.state == "CLOSED":
                return True
            
            if self.state == "OPEN":
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self.state = "HALF_OPEN"
                        logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                        return True
                return False
            
            # HALF_OPEN - allow one test call
            return True
    
    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self.failures = 0
            if self.state != "CLOSED":
                logger.info(f"Circuit breaker '{self.name}' recovered, entering CLOSED state")
            self.state = "CLOSED"
    
    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.now()
            
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                logger.warning(f"Circuit breaker '{self.name}' test failed, returning to OPEN state")
            elif self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN after {self.failures} failures. "
                    f"Will retry in {self.recovery_timeout}s"
                )


# Global circuit breakers for different services
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _circuit_breakers[name]


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    circuit_breaker_name: Optional[str] = None
):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry
        circuit_breaker_name: Optional circuit breaker to use
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            circuit_breaker = None
            if circuit_breaker_name:
                circuit_breaker = get_circuit_breaker(circuit_breaker_name)
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                # Check circuit breaker
                if circuit_breaker and not circuit_breaker.can_execute():
                    raise RuntimeError(
                        f"Circuit breaker '{circuit_breaker_name}' is OPEN. "
                        f"Service appears to be down."
                    )
                
                try:
                    result = func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result
                    
                except retryable_exceptions as e:
                    last_exception = e
                    category = ErrorClassifier.classify(e)
                    
                    # Don't retry permanent errors
                    if category == ErrorCategory.PERMANENT:
                        logger.warning(f"Permanent error in {func.__name__}: {e}")
                        raise
                    
                    # Record failure for circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    # Check if we have retries left
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    default_value: T = None,
    log_errors: bool = True,
    error_message: str = None,
    reraise: bool = False
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that catches exceptions and returns a default value.
    
    Args:
        default_value: Value to return on error
        log_errors: Whether to log errors
        error_message: Custom error message prefix
        reraise: Whether to re-raise after logging
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    msg = error_message or f"Error in {func.__name__}"
                    logger.error(f"{msg}: {e}", exc_info=True)
                
                if reraise:
                    raise
                
                return default_value
        
        return wrapper
    return decorator


class ErrorAggregator:
    """
    Aggregates errors during a pipeline run for reporting.
    
    Usage:
        aggregator = ErrorAggregator()
        with aggregator.context("processing_episode", episode_id=episode.id):
            # ... do work ...
            # errors are automatically recorded
        
        print(aggregator.get_summary())
    """
    
    def __init__(self, max_errors: int = 1000):
        self.errors: List[ErrorRecord] = []
        self.max_errors = max_errors
        self._lock = threading.Lock()
        self._context_stack: List[Dict[str, Any]] = []
    
    def record(self, 
               error: Exception, 
               context: Dict[str, Any] = None,
               include_traceback: bool = True):
        """Record an error."""
        with self._lock:
            if len(self.errors) >= self.max_errors:
                return  # Prevent memory issues
            
            # Merge context from stack
            full_context = {}
            for ctx in self._context_stack:
                full_context.update(ctx)
            if context:
                full_context.update(context)
            
            record = ErrorRecord(
                error_type=type(error).__name__,
                message=str(error),
                category=ErrorClassifier.classify(error),
                timestamp=datetime.now(),
                context=full_context,
                traceback=traceback.format_exc() if include_traceback else None
            )
            
            self.errors.append(record)
    
    def push_context(self, **kwargs):
        """Push context onto the stack."""
        self._context_stack.append(kwargs)
    
    def pop_context(self):
        """Pop context from the stack."""
        if self._context_stack:
            self._context_stack.pop()
    
    class context:
        """Context manager for error context."""
        def __init__(self, aggregator: 'ErrorAggregator', name: str, **kwargs):
            self.aggregator = aggregator
            self.context_data = {"operation": name, **kwargs}
        
        def __enter__(self):
            self.aggregator.push_context(**self.context_data)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val:
                self.aggregator.record(exc_val)
            self.aggregator.pop_context()
            return False  # Don't suppress exceptions
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        with self._lock:
            by_category = {}
            by_type = {}
            
            for error in self.errors:
                cat = error.category.value
                by_category[cat] = by_category.get(cat, 0) + 1
                
                by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
            
            return {
                "total_errors": len(self.errors),
                "by_category": by_category,
                "by_type": by_type,
                "first_error": self.errors[0].timestamp.isoformat() if self.errors else None,
                "last_error": self.errors[-1].timestamp.isoformat() if self.errors else None
            }
    
    def get_errors_for_context(self, **context_filter) -> List[ErrorRecord]:
        """Get errors matching a context filter."""
        with self._lock:
            results = []
            for error in self.errors:
                matches = all(
                    error.context.get(k) == v 
                    for k, v in context_filter.items()
                )
                if matches:
                    results.append(error)
            return results
    
    def clear(self):
        """Clear all recorded errors."""
        with self._lock:
            self.errors.clear()


class ResilientPipeline:
    """
    Base class for resilient pipeline execution with checkpointing.
    
    Subclass this to create pipelines that can resume from failures.
    """
    
    def __init__(self, checkpoint_file: str = None):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data: Dict[str, Any] = {}
        self.error_aggregator = ErrorAggregator()
        
        # Load existing checkpoint
        if checkpoint_file:
            self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint from file."""
        import json
        from pathlib import Path
        
        path = Path(self.checkpoint_file)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.checkpoint_data = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
    
    def _save_checkpoint(self):
        """Save checkpoint to file."""
        import json
        
        if not self.checkpoint_file:
            return
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def mark_completed(self, item_id: str, result: Any = None):
        """Mark an item as completed."""
        if "completed" not in self.checkpoint_data:
            self.checkpoint_data["completed"] = {}
        
        self.checkpoint_data["completed"][item_id] = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        self._save_checkpoint()
    
    def is_completed(self, item_id: str) -> bool:
        """Check if an item has been completed."""
        return item_id in self.checkpoint_data.get("completed", {})
    
    def get_completed_items(self) -> List[str]:
        """Get list of completed item IDs."""
        return list(self.checkpoint_data.get("completed", {}).keys())


# Convenience function for wrapping risky operations
def try_or_default(operation: Callable[[], T], default: T = None, log: bool = True) -> T:
    """
    Execute an operation and return default on any error.
    
    Usage:
        result = try_or_default(lambda: risky_operation(), default="fallback")
    """
    try:
        return operation()
    except Exception as e:
        if log:
            logger.warning(f"Operation failed, using default: {e}")
        return default


def try_multiple(operations: List[Callable[[], T]], 
                 operation_names: List[str] = None) -> Tuple[Optional[T], Optional[str]]:
    """
    Try multiple operations in sequence, return first success.
    
    Usage:
        result, method = try_multiple([
            lambda: method1(),
            lambda: method2(),
            lambda: method3(),
        ], ["YouTube", "LocalWhisper", "WhisperAPI"])
    
    Returns:
        Tuple of (result, method_name) or (None, None) if all failed
    """
    names = operation_names or [f"method_{i}" for i in range(len(operations))]
    
    for op, name in zip(operations, names):
        try:
            result = op()
            if result is not None:
                logger.info(f"Success with {name}")
                return result, name
        except Exception as e:
            logger.debug(f"Failed with {name}: {e}")
            continue
    
    return None, None







