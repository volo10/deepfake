"""
Parallel processing module for the Deepfake Detection Agent.

This module provides utilities for:
- Parallel video processing using multiprocessing
- Async I/O for batch operations
- Thread-safe result aggregation
- Progress tracking

Usage:
    from deepfake_detector.parallel import ParallelAnalyzer
    
    analyzer = ParallelAnalyzer(num_workers=4)
    results = analyzer.analyze_batch(video_paths)
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    Future,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from .models import DetectionResult, AnalysisConfig

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class BatchProgress:
    """
    Thread-safe progress tracker for batch operations.
    
    Attributes:
        total: Total number of items to process
        completed: Number of completed items
        failed: Number of failed items
        current_video: Currently processing video
    """
    
    total: int
    completed: int = 0
    failed: int = 0
    current_video: str = ""
    start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, success: bool, video: str = "") -> None:
        """Thread-safe progress update."""
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            self.current_video = video
    
    @property
    def progress_pct(self) -> float:
        """Return progress percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed + self.failed) / self.total * 100
    
    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining(self) -> float | None:
        """Estimate remaining time in seconds."""
        done = self.completed + self.failed
        if done == 0:
            return None
        rate = done / self.elapsed_seconds
        remaining = self.total - done
        return remaining / rate if rate > 0 else None
    
    def __str__(self) -> str:
        """Return progress string."""
        done = self.completed + self.failed
        eta = self.estimated_remaining
        eta_str = f", ETA: {eta:.0f}s" if eta else ""
        return f"Progress: {done}/{self.total} ({self.progress_pct:.1f}%){eta_str}"


@dataclass
class BatchResult:
    """
    Results from batch processing.
    
    Attributes:
        results: List of (video_path, result_or_error) tuples
        total_time: Total processing time
        success_count: Number of successful analyses
        failure_count: Number of failed analyses
    """
    
    results: list[tuple[str, DetectionResult | Exception]]
    total_time: float
    success_count: int
    failure_count: int
    
    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        total = self.success_count + self.failure_count
        return self.success_count / total * 100 if total > 0 else 0.0
    
    def get_successful(self) -> list[tuple[str, DetectionResult]]:
        """Return only successful results."""
        return [
            (path, result)
            for path, result in self.results
            if isinstance(result, DetectionResult)
        ]
    
    def get_failed(self) -> list[tuple[str, Exception]]:
        """Return only failed results."""
        return [
            (path, result)
            for path, result in self.results
            if isinstance(result, Exception)
        ]


class ParallelAnalyzer:
    """
    Parallel video analyzer using multiprocessing.
    
    Uses ProcessPoolExecutor for CPU-bound video analysis
    and ThreadPoolExecutor for I/O operations.
    
    Example:
        >>> analyzer = ParallelAnalyzer(num_workers=4)
        >>> results = analyzer.analyze_batch(
        ...     ["video1.mp4", "video2.mp4", "video3.mp4"],
        ...     progress_callback=print_progress
        ... )
        >>> print(f"Success rate: {results.success_rate:.1f}%")
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        config: AnalysisConfig | None = None,
        use_processes: bool = True
    ) -> None:
        """
        Initialize the parallel analyzer.
        
        Args:
            num_workers: Number of parallel workers
            config: Analysis configuration to use
            use_processes: If True, use ProcessPoolExecutor (recommended).
                          If False, use ThreadPoolExecutor.
        """
        self.num_workers = num_workers
        self.config = config or AnalysisConfig()
        self.use_processes = use_processes
        self._progress: BatchProgress | None = None
    
    def analyze_batch(
        self,
        video_paths: list[str | Path],
        progress_callback: Callable[[BatchProgress], None] | None = None,
        error_callback: Callable[[str, Exception], None] | None = None
    ) -> BatchResult:
        """
        Analyze multiple videos in parallel.
        
        Args:
            video_paths: List of paths to video files
            progress_callback: Optional callback for progress updates
            error_callback: Optional callback for error handling
        
        Returns:
            BatchResult containing all results and statistics
        
        Example:
            >>> def on_progress(p):
            ...     print(f"\\r{p}", end="")
            >>> results = analyzer.analyze_batch(videos, progress_callback=on_progress)
        """
        start_time = time.time()
        video_paths = [str(p) for p in video_paths]
        
        self._progress = BatchProgress(total=len(video_paths))
        results: list[tuple[str, DetectionResult | Exception]] = []
        
        # Use appropriate executor
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_video: dict[Future, str] = {}
            
            for video_path in video_paths:
                future = executor.submit(self._analyze_single, video_path)
                future_to_video[future] = video_path
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                
                try:
                    result = future.result()
                    results.append((video_path, result))
                    self._progress.update(success=True, video=video_path)
                    
                except Exception as e:
                    results.append((video_path, e))
                    self._progress.update(success=False, video=video_path)
                    
                    if error_callback:
                        error_callback(video_path, e)
                    
                    logger.error(f"Failed to analyze {video_path}: {e}")
                
                if progress_callback:
                    progress_callback(self._progress)
        
        total_time = time.time() - start_time
        
        return BatchResult(
            results=results,
            total_time=total_time,
            success_count=self._progress.completed,
            failure_count=self._progress.failed
        )
    
    def _analyze_single(self, video_path: str) -> DetectionResult:
        """
        Analyze a single video (runs in worker process).
        
        Note: This method is called in a separate process,
        so it must re-import and re-initialize the detector.
        """
        # Import here to avoid pickling issues
        from .detector import DeepfakeDetector
        
        detector = DeepfakeDetector(config=self.config)
        return detector.analyze(video_path)
    
    def analyze_streaming(
        self,
        video_paths: list[str | Path]
    ) -> Iterator[tuple[str, DetectionResult | Exception]]:
        """
        Analyze videos and yield results as they complete.
        
        This is useful for processing large batches where you
        want to handle results incrementally.
        
        Args:
            video_paths: List of paths to video files
        
        Yields:
            Tuples of (video_path, result_or_exception)
        
        Example:
            >>> for path, result in analyzer.analyze_streaming(videos):
            ...     if isinstance(result, DetectionResult):
            ...         print(f"{path}: {result.verdict}")
            ...     else:
            ...         print(f"{path}: ERROR - {result}")
        """
        video_paths = [str(p) for p in video_paths]
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            future_to_video: dict[Future, str] = {}
            
            for video_path in video_paths:
                future = executor.submit(self._analyze_single, video_path)
                future_to_video[future] = video_path
            
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                
                try:
                    result = future.result()
                    yield (video_path, result)
                except Exception as e:
                    yield (video_path, e)


def analyze_video_worker(args: tuple[str, dict[str, Any]]) -> tuple[str, DetectionResult]:
    """
    Worker function for multiprocessing.
    
    This is a module-level function for use with multiprocessing.Pool.
    
    Args:
        args: Tuple of (video_path, config_dict)
    
    Returns:
        Tuple of (video_path, result)
    """
    video_path, config_dict = args
    
    from .detector import DeepfakeDetector
    from .models import AnalysisConfig
    
    config = AnalysisConfig(**config_dict) if config_dict else AnalysisConfig()
    detector = DeepfakeDetector(config=config)
    result = detector.analyze(video_path)
    
    return (video_path, result)


class ThreadSafeCounter:
    """Thread-safe counter for tracking metrics."""
    
    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= amount
            return self._value
    
    @property
    def value(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value


class RateLimiter:
    """
    Simple rate limiter for controlling processing speed.
    
    Useful when you need to limit API calls or resource usage.
    """
    
    def __init__(self, max_per_second: float) -> None:
        """
        Initialize rate limiter.
        
        Args:
            max_per_second: Maximum operations per second
        """
        self.min_interval = 1.0 / max_per_second
        self.last_time = 0.0
        self._lock = threading.Lock()
    
    def acquire(self) -> None:
        """
        Wait until rate limit allows another operation.
        
        This method is thread-safe.
        """
        with self._lock:
            current = time.time()
            elapsed = current - self.last_time
            
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            
            self.last_time = time.time()


# Convenience function for simple batch processing
def analyze_batch(
    video_paths: list[str | Path],
    num_workers: int = 4,
    config: AnalysisConfig | None = None,
    progress: bool = True
) -> BatchResult:
    """
    Convenience function for batch video analysis.
    
    Args:
        video_paths: List of video file paths
        num_workers: Number of parallel workers
        config: Optional analysis configuration
        progress: Whether to print progress
    
    Returns:
        BatchResult with all results
    
    Example:
        >>> results = analyze_batch(["v1.mp4", "v2.mp4", "v3.mp4"])
        >>> for path, result in results.get_successful():
        ...     print(f"{path}: {result.verdict}")
    """
    def progress_callback(p: BatchProgress) -> None:
        if progress:
            print(f"\r{p}", end="", flush=True)
    
    analyzer = ParallelAnalyzer(num_workers=num_workers, config=config)
    result = analyzer.analyze_batch(
        video_paths,
        progress_callback=progress_callback if progress else None
    )
    
    if progress:
        print()  # New line after progress
    
    return result

