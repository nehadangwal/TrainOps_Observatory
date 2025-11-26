"""
Bottleneck Detection and Analysis Engine
Identifies training inefficiencies and provides optimization recommendations
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from sqlalchemy import func

from app.models import db, Run, Metric


class BottleneckType(Enum):
    """Types of bottlenecks detected"""
    IO_BOUND = "io_bound"
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    OVERSIZED_INSTANCE = "oversized_instance"
    UNDERSIZED_BATCH = "undersized_batch"
    NO_BOTTLENECK = "no_bottleneck"


class Severity(Enum):
    """Severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Bottleneck:
    """Detected bottleneck with recommendation"""
    type: BottleneckType
    severity: Severity
    description: str
    recommendation: str
    estimated_speedup: Optional[str] = None
    estimated_savings: Optional[float] = None
    confidence: float = 0.0  # 0-1 scale
    
    def to_dict(self) -> Dict:
        return {
            'type': self.type.value,
            'severity': self.severity.value,
            'description': self.description,
            'recommendation': self.recommendation,
            'estimated_speedup': self.estimated_speedup,
            'estimated_savings': self.estimated_savings,
            'confidence': round(self.confidence, 2)
        }


@dataclass
class MetricsSummary:
    """Statistical summary of metrics"""
    avg_gpu_util: float
    avg_cpu_util: float
    avg_throughput: float
    max_gpu_memory_util: float
    avg_gpu_memory_util: float
    min_gpu_util: float
    max_gpu_util: float
    std_gpu_util: float
    data_points: int


class BottleneckAnalyzer:
    """
    Analyzes training runs to identify bottlenecks and optimization opportunities
    """
    
    # Thresholds for detection
    LOW_GPU_UTIL_THRESHOLD = 70  # %
    HIGH_CPU_UTIL_THRESHOLD = 85  # %
    HIGH_GPU_MEMORY_THRESHOLD = 95  # %
    LOW_GPU_UTIL_CRITICAL = 50  # %
    UNDERSIZED_BATCH_THROUGHPUT = 50  # samples/sec
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run = Run.query.get(run_id)
        if not self.run:
            raise ValueError(f"Run {run_id} not found")
        
        self.metrics_summary = self._calculate_summary()
    
    def _calculate_summary(self) -> MetricsSummary:
        """Calculate statistical summary of metrics"""
        metrics = Metric.query.filter_by(run_id=self.run_id).all()
        
        if not metrics:
            raise ValueError(f"No metrics found for run {self.run_id}")
        
        # Extract values
        gpu_utils = [m.gpu_util for m in metrics if m.gpu_util is not None]
        cpu_utils = [m.cpu_util for m in metrics if m.cpu_util is not None]
        throughputs = [m.throughput for m in metrics if m.throughput is not None]
        gpu_mem_utils = [m.gpu_memory_util for m in metrics if m.gpu_memory_util is not None]
        
        return MetricsSummary(
            avg_gpu_util=statistics.mean(gpu_utils) if gpu_utils else 0,
            avg_cpu_util=statistics.mean(cpu_utils) if cpu_utils else 0,
            avg_throughput=statistics.mean(throughputs) if throughputs else 0,
            max_gpu_memory_util=max(gpu_mem_utils) if gpu_mem_utils else 0,
            avg_gpu_memory_util=statistics.mean(gpu_mem_utils) if gpu_mem_utils else 0,
            min_gpu_util=min(gpu_utils) if gpu_utils else 0,
            max_gpu_util=max(gpu_utils) if gpu_utils else 0,
            std_gpu_util=statistics.stdev(gpu_utils) if len(gpu_utils) > 1 else 0,
            data_points=len(metrics)
        )
    
    def analyze(self) -> List[Bottleneck]:
        """
        Run all bottleneck detection algorithms
        Returns list of detected bottlenecks ordered by severity
        """
        bottlenecks = []
        
        # Run detection algorithms
        bottlenecks.extend(self._detect_io_bound())
        bottlenecks.extend(self._detect_cpu_bound())
        bottlenecks.extend(self._detect_memory_bound())
        bottlenecks.extend(self._detect_oversized_instance())
        bottlenecks.extend(self._detect_undersized_batch())
        
        # If no bottlenecks found, add success message
        if not bottlenecks:
            bottlenecks.append(Bottleneck(
                type=BottleneckType.NO_BOTTLENECK,
                severity=Severity.INFO,
                description="Training appears well-optimized",
                recommendation="GPU utilization is healthy. Consider monitoring over time.",
                confidence=0.8
            ))
        
        # Sort by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4
        }
        bottlenecks.sort(key=lambda x: (severity_order[x.severity], -x.confidence))
        
        return bottlenecks
    
    def _detect_io_bound(self) -> List[Bottleneck]:
        """Detect I/O bottlenecks (data loading issues)"""
        bottlenecks = []
        summary = self.metrics_summary
        
        # Rule: Low GPU utilization indicates data loading bottleneck
        if summary.avg_gpu_util < self.LOW_GPU_UTIL_THRESHOLD:
            # Calculate confidence based on how low GPU util is
            gpu_util_gap = self.LOW_GPU_UTIL_THRESHOLD - summary.avg_gpu_util
            confidence = min(gpu_util_gap / self.LOW_GPU_UTIL_THRESHOLD, 1.0)
            
            # Determine severity
            if summary.avg_gpu_util < self.LOW_GPU_UTIL_CRITICAL:
                severity = Severity.CRITICAL
            else:
                severity = Severity.HIGH
            
            # Estimate speedup potential
            target_util = 85
            potential_speedup = target_util / max(summary.avg_gpu_util, 1)
            speedup_str = f"{potential_speedup:.1f}x" if potential_speedup < 5 else "3-5x"
            
            bottlenecks.append(Bottleneck(
                type=BottleneckType.IO_BOUND,
                severity=severity,
                description=f"GPU underutilized at {summary.avg_gpu_util:.1f}% (target: 80%+). "
                           f"Training likely waiting for data loading.",
                recommendation="Optimize data loading:\n"
                             "  • Add num_workers=4 (or higher) to DataLoader\n"
                             "  • Enable pin_memory=True for GPU training\n"
                             "  • Consider data prefetching or caching\n"
                             "  • Profile data preprocessing pipeline",
                estimated_speedup=speedup_str,
                confidence=confidence
            ))
        
        return bottlenecks
    
    def _detect_cpu_bound(self) -> List[Bottleneck]:
        """Detect CPU bottlenecks (preprocessing issues)"""
        bottlenecks = []
        summary = self.metrics_summary
        
        # Rule: High CPU + Low GPU indicates CPU preprocessing bottleneck
        if (summary.avg_cpu_util > self.HIGH_CPU_UTIL_THRESHOLD and 
            summary.avg_gpu_util < self.LOW_GPU_UTIL_THRESHOLD):
            
            confidence = min(
                (summary.avg_cpu_util - self.HIGH_CPU_UTIL_THRESHOLD) / 15,
                0.9
            )
            
            bottlenecks.append(Bottleneck(
                type=BottleneckType.CPU_BOUND,
                severity=Severity.HIGH,
                description=f"CPU at {summary.avg_cpu_util:.1f}% while GPU at {summary.avg_gpu_util:.1f}%. "
                           f"Data preprocessing is the bottleneck.",
                recommendation="Optimize preprocessing:\n"
                             "  • Move transforms to GPU (e.g., Kornia for PyTorch)\n"
                             "  • Reduce data augmentation complexity\n"
                             "  • Pre-compute expensive transformations\n"
                             "  • Use faster image libraries (e.g., libjpeg-turbo)",
                estimated_speedup="1.5-2x",
                confidence=confidence
            ))
        
        return bottlenecks
    
    def _detect_memory_bound(self) -> List[Bottleneck]:
        """Detect GPU memory bottlenecks"""
        bottlenecks = []
        summary = self.metrics_summary
        
        # Rule: High GPU memory usage risks OOM errors
        if summary.max_gpu_memory_util > self.HIGH_GPU_MEMORY_THRESHOLD:
            bottlenecks.append(Bottleneck(
                type=BottleneckType.MEMORY_BOUND,
                severity=Severity.CRITICAL,
                description=f"GPU memory at {summary.max_gpu_memory_util:.1f}% (critical threshold: 95%). "
                           f"High risk of Out-of-Memory errors.",
                recommendation="Reduce memory usage:\n"
                             "  • Reduce batch size\n"
                             "  • Use gradient accumulation for effective larger batches\n"
                             "  • Enable gradient checkpointing\n"
                             "  • Use mixed precision training (fp16/bf16)\n"
                             "  • Consider model parallelism for large models",
                estimated_speedup="N/A (prevents crashes)",
                confidence=0.95
            ))
        
        return bottlenecks
    
    def _detect_oversized_instance(self) -> List[Bottleneck]:
        """Detect when instance is oversized for workload"""
        bottlenecks = []
        summary = self.metrics_summary
        
        # Rule: Consistently low GPU utilization suggests oversized instance
        if (summary.avg_gpu_util < 50 and 
            summary.max_gpu_util < 60 and
            summary.std_gpu_util < 10):  # Consistently low
            
            # Estimate potential savings
            util_ratio = summary.avg_gpu_util / 85  # Target 85% util
            potential_savings_pct = int((1 - util_ratio) * 100)
            
            bottlenecks.append(Bottleneck(
                type=BottleneckType.OVERSIZED_INSTANCE,
                severity=Severity.MEDIUM,
                description=f"GPU consistently underutilized (avg: {summary.avg_gpu_util:.1f}%, "
                           f"max: {summary.max_gpu_util:.1f}%). Instance may be oversized.",
                recommendation="Consider right-sizing:\n"
                             "  • Try smaller/cheaper GPU instance\n"
                             "  • First fix data loading bottleneck if present\n"
                             "  • Profile to confirm GPU isn't needed\n"
                             f"  • Potential cost savings: ~{potential_savings_pct}%",
                estimated_speedup=None,
                confidence=0.7
            ))
        
        return bottlenecks
    
    def _detect_undersized_batch(self) -> List[Bottleneck]:
        """Detect when batch size is too small"""
        bottlenecks = []
        summary = self.metrics_summary
        
        # Rule: Low throughput with low GPU memory suggests small batches
        if (summary.avg_throughput < self.UNDERSIZED_BATCH_THROUGHPUT and
            summary.avg_gpu_memory_util < 50):
            
            bottlenecks.append(Bottleneck(
                type=BottleneckType.UNDERSIZED_BATCH,
                severity=Severity.MEDIUM,
                description=f"Low throughput ({summary.avg_throughput:.1f} samples/sec) with "
                           f"low GPU memory usage ({summary.avg_gpu_memory_util:.1f}%). "
                           f"Batch size may be too small.",
                recommendation="Increase batch size:\n"
                             "  • Try doubling batch size\n"
                             "  • Monitor GPU memory usage\n"
                             "  • Use gradient accumulation if memory limited\n"
                             "  • Larger batches improve GPU utilization",
                estimated_speedup="1.5-2x",
                confidence=0.6
            ))
        
        return bottlenecks
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive analysis report
        """
        bottlenecks = self.analyze()
        
        # Calculate potential impact
        total_savings_potential = sum(
            b.estimated_savings for b in bottlenecks 
            if b.estimated_savings
        )
        
        # Priority recommendations
        high_priority = [
            b for b in bottlenecks 
            if b.severity in [Severity.CRITICAL, Severity.HIGH]
        ]
        
        return {
            'run_id': self.run_id,
            'run_name': self.run.name,
            'summary': {
                'avg_gpu_util': round(self.metrics_summary.avg_gpu_util, 1),
                'avg_cpu_util': round(self.metrics_summary.avg_cpu_util, 1),
                'avg_throughput': round(self.metrics_summary.avg_throughput, 1),
                'max_gpu_memory': round(self.metrics_summary.max_gpu_memory_util, 1),
                'data_points': self.metrics_summary.data_points
            },
            'bottlenecks': [b.to_dict() for b in bottlenecks],
            'bottleneck_count': len(bottlenecks),
            'high_priority_count': len(high_priority),
            'estimated_total_savings': round(total_savings_potential, 2) if total_savings_potential else None,
            'recommendations': [
                {
                    'priority': b.severity.value,
                    'action': b.recommendation.split('\n')[0],
                    'impact': b.estimated_speedup or 'N/A'
                }
                for b in high_priority[:3]  # Top 3 recommendations
            ]
        }
