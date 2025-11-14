"""
TrainOps - Lightweight observability for ML training workflows
"""

__version__ = "0.1.0"

from trainops.monitor import TrainOpsMonitor, MetricsCollector

__all__ = ["TrainOpsMonitor", "MetricsCollector"]
