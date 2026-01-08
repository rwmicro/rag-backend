"""
GPU Queue Manager for managing GPU workload distribution
"""

import threading
from queue import Queue
from typing import Optional, Callable, Any
from loguru import logger


class GPUQueueManager:
    """
    Manager for queuing GPU-intensive tasks to prevent overload
    """

    def __init__(self, max_workers: int = 1):
        """
        Initialize GPU queue manager

        Args:
            max_workers: Maximum number of concurrent GPU tasks
        """
        self.max_workers = max_workers
        self.queue: Queue = Queue()
        self.workers = []
        self.running = False
        logger.info(f"Initialized GPUQueueManager with {max_workers} workers")

    def start(self):
        """Start worker threads"""
        if self.running:
            return

        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
            logger.debug(f"Started GPU worker thread {i}")

    def stop(self):
        """Stop all worker threads"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)
        self.workers.clear()
        logger.info("Stopped all GPU workers")

    def _worker(self):
        """Worker thread that processes tasks from the queue"""
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    break

                func, args, kwargs, callback = task
                try:
                    result = func(*args, **kwargs)
                    if callback:
                        callback(result)
                except Exception as e:
                    logger.error(f"Error in GPU task: {e}")
                finally:
                    self.queue.task_done()
            except:
                continue

    def submit(
        self,
        func: Callable,
        *args,
        callback: Optional[Callable[[Any], None]] = None,
        **kwargs,
    ) -> None:
        """
        Submit a task to the GPU queue

        Args:
            func: Function to execute
            *args: Positional arguments for func
            callback: Optional callback function to call with result
            **kwargs: Keyword arguments for func
        """
        if not self.running:
            self.start()

        self.queue.put((func, args, kwargs, callback))
        logger.debug(f"Submitted task {func.__name__} to GPU queue")

    def wait(self):
        """Wait for all tasks in queue to complete"""
        self.queue.join()


# Global GPU queue manager instance
_gpu_manager: Optional[GPUQueueManager] = None


def get_gpu_manager(max_workers: int = 1) -> GPUQueueManager:
    """
    Get or create the global GPU queue manager

    Args:
        max_workers: Maximum concurrent GPU tasks

    Returns:
        GPUQueueManager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUQueueManager(max_workers=max_workers)
        _gpu_manager.start()
    return _gpu_manager
