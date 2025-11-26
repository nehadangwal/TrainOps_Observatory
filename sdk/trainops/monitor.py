"""
TrainOps Monitor - Core instrumentation for ML training workflows
"""
import time
import functools
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from uuid import uuid4

import requests
import psutil

from trainops.config import Config, logger

try:
    import pynvml
    NVML_AVAILABLE = True and Config.ENABLE_GPU_METRICS
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available, GPU metrics will be disabled")


class MetricsCollector:
    """Collects system and training metrics with minimal overhead"""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_count = 0
        self.last_throughput_time = time.time()
        self.last_throughput_step = 0
        
        # Initialize NVML for GPU metrics
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
            except Exception as e:
                logger.warning(f"GPU metrics unavailable: {e}")
                self.gpu_available = False
        else:
            self.gpu_available = False
    
    def collect(self) -> Dict[str, Any]:
        """Collect current metrics snapshot"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
        }
        
        # System metrics
        metrics['cpu_util'] = psutil.cpu_percent(interval=0.1)
        metrics['system_memory'] = psutil.virtual_memory().percent
        
        # I/O metrics
        io_counters = psutil.disk_io_counters()
        if io_counters:
            metrics['io_read_mb'] = io_counters.read_bytes / (1024 * 1024)
            metrics['io_write_mb'] = io_counters.write_bytes / (1024 * 1024)
        
        # GPU metrics
        if self.gpu_available:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                
                metrics['gpu_util'] = util.gpu
                metrics['gpu_memory_used'] = mem.used / (1024 ** 3)  # GB
                metrics['gpu_memory_total'] = mem.total / (1024 ** 3)  # GB
                metrics['gpu_memory_util'] = (mem.used / mem.total) * 100
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
        
        # Throughput (samples/sec)
        current_time = time.time()
        time_delta = current_time - self.last_throughput_time
        if time_delta > 0:
            step_delta = self.step_count - self.last_throughput_step
            metrics['throughput'] = step_delta / time_delta
        
        return metrics
    
    def update_step(self):
        """Called after each training step"""
        self.step_count += 1
    
    def reset_throughput(self):
        """Reset throughput calculation"""
        self.last_throughput_time = time.time()
        self.last_throughput_step = self.step_count


class TrainOpsMonitor:
    """
    Main monitor class for instrumenting training workflows
    
    Example:
        monitor = TrainOpsMonitor(
            run_name="resnet50_baseline",
            project="image_classification"
        )
        
        @monitor.track_training
        def train_epoch(model, dataloader, optimizer):
            for batch in dataloader:
                # ... training code
                monitor.log_step(loss=loss.item())
    """
    
    def __init__(
        self,
        run_name: str,
        project: str = "default",
        api_url: str = "http://localhost:5000",
        instance_type: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        collect_interval: int = 10,  # seconds
        send_interval: int = 30,  # seconds
    ):
        self.run_name = run_name
        self.project = project
        self.api_url = api_url.rstrip('/')
        self.instance_type = instance_type or self._detect_instance_type()
        self.tags = tags or {}
        self.collect_interval = collect_interval
        self.send_interval = send_interval
        
        # State
        self.run_id = None
        self.collector = MetricsCollector()
        self.metrics_buffer = []
        self.custom_metrics = {}
        self.last_send_time = time.time()
        self.epoch_num = 0
        
        # Threading for background collection
        self.collecting = False
        self.collection_thread = None
        
        # Create run
        self._create_run()
    
    def _detect_instance_type(self) -> str:
        """Attempt to detect cloud instance type"""
        # TODO: Add AWS/GCP metadata detection
        return "unknown"
    
    def _create_run(self):
        """Register new training run with backend"""
        try:
            response = requests.post(
                f"{self.api_url}/api/runs",
                json={
                    'name': self.run_name,
                    'project': self.project,
                    'instance_type': self.instance_type,
                    'tags': self.tags,
                    'started_at': datetime.now().isoformat(),
                },
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            self.run_id = data['run_id']
            logger.info(f"Created run: {self.run_id}")
        except Exception as e:
            logger.error(f"Failed to create run: {e}")
            # Generate local ID and continue
            self.run_id = str(uuid4())
            logger.warning(f"Using local run ID: {self.run_id}")
    
    def _collect_metrics_loop(self):
        """Background thread for collecting metrics"""
        while self.collecting:
            try:
                metrics = self.collector.collect()
                
                # Add custom metrics
                if self.custom_metrics:
                    metrics['custom'] = self.custom_metrics.copy()
                    self.custom_metrics.clear()
                
                self.metrics_buffer.append(metrics)
                
                # Send if buffer is full or interval elapsed
                if self._should_send():
                    self._send_metrics()
                
                # Reset throughput calculation periodically
                self.collector.reset_throughput()
                
            except Exception as e:
                logger.warning(f"Error collecting metrics: {e}")
            
            time.sleep(self.collect_interval)
    
    def _should_send(self) -> bool:
        """Check if we should send buffered metrics"""
        return (
            len(self.metrics_buffer) >= 50 or
            time.time() - self.last_send_time > self.send_interval
        )
    
    def _send_metrics(self):
        """Send buffered metrics to backend"""
        if not self.metrics_buffer:
            return
        
        try:
            response = requests.post(
                f"{self.api_url}/api/runs/{self.run_id}/metrics",
                json={'metrics': self.metrics_buffer},
                timeout=5
            )
            response.raise_for_status()
            
            logger.debug(f"Sent {len(self.metrics_buffer)} metrics")
            self.metrics_buffer = []
            self.last_send_time = time.time()
            
        except Exception as e:
            logger.warning(f"Failed to send metrics: {e}")
            # Keep buffer for retry, but limit size
            if len(self.metrics_buffer) > 500:
                self.metrics_buffer = self.metrics_buffer[-500:]
    
    def start_collection(self):
        """Start background metrics collection"""
        if self.collecting:
            return
        
        self.collecting = True
        self.collection_thread = threading.Thread(
            target=self._collect_metrics_loop,
            daemon=True
        )
        self.collection_thread.start()
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop background metrics collection"""
        if not self.collecting:
            return
        
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        # Send remaining metrics
        self._send_metrics()
        logger.info("Stopped metrics collection")
    
    def log_step(self, **kwargs):
        """
        Log custom metrics for current step
        
        Args:
            **kwargs: Arbitrary metrics (loss=0.5, accuracy=0.95, etc.)
        """
        self.custom_metrics.update(kwargs)
        self.collector.update_step()
    
    def log_epoch(self, epoch: int):
        """Mark end of epoch"""
        self.epoch_num = epoch
        logger.info(f"Completed epoch {epoch}")
    
    def track_training(self, func: Callable) -> Callable:
        """
        Decorator to automatically track training function
        
        Example:
            @monitor.track_training
            def train_epoch(model, dataloader):
                for batch in dataloader:
                    # ... training code
                    monitor.log_step(loss=loss.item())
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.run_id:
                return func(*args, **kwargs) # Skip if setup failed
            
            self.start_collection() # Start collection only once (due to internal checks)
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                # Log the error, but the explicit monitor.finish() will handle cleanup
                raise
        
        return wrapper
    
    def finish(self):
        """Mark run as complete"""
        self.stop_collection()
        
        try:
            response = requests.patch(
                f"{self.api_url}/api/runs/{self.run_id}",
                json={
                    'status': 'completed',
                    'ended_at': datetime.now().isoformat(),
                },
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"Finished run: {self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to mark run as finished: {e}")
    
    def __enter__(self):
        """Context manager support"""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.finish()
        return False