"""
Cost Estimation Engine
Calculates training costs and identifies optimization savings
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from app.models import Run


# AWS EC2 pricing (as of November 2024, us-east-1 on-demand)
AWS_PRICING = {
    # GPU Instances
    'p3.2xlarge': {'gpu': 'V100', 'gpu_count': 1, 'cost_per_hour': 3.06},
    'p3.8xlarge': {'gpu': '4x V100', 'gpu_count': 4, 'cost_per_hour': 12.24},
    'p3.16xlarge': {'gpu': '8x V100', 'gpu_count': 8, 'cost_per_hour': 24.48},
    'p3dn.24xlarge': {'gpu': '8x V100', 'gpu_count': 8, 'cost_per_hour': 31.22},
    
    'p4d.24xlarge': {'gpu': '8x A100', 'gpu_count': 8, 'cost_per_hour': 32.77},
    'p4de.24xlarge': {'gpu': '8x A100 80GB', 'gpu_count': 8, 'cost_per_hour': 40.97},
    
    'p5.48xlarge': {'gpu': '8x H100', 'gpu_count': 8, 'cost_per_hour': 98.32},
    
    'g4dn.xlarge': {'gpu': 'T4', 'gpu_count': 1, 'cost_per_hour': 0.526},
    'g4dn.2xlarge': {'gpu': 'T4', 'gpu_count': 1, 'cost_per_hour': 0.752},
    'g4dn.12xlarge': {'gpu': '4x T4', 'gpu_count': 4, 'cost_per_hour': 3.912},
    
    'g5.xlarge': {'gpu': 'A10G', 'gpu_count': 1, 'cost_per_hour': 1.006},
    'g5.2xlarge': {'gpu': 'A10G', 'gpu_count': 1, 'cost_per_hour': 1.212},
    'g5.12xlarge': {'gpu': '4x A10G', 'gpu_count': 4, 'cost_per_hour': 5.672},
    'g5.48xlarge': {'gpu': '8x A10G', 'gpu_count': 8, 'cost_per_hour': 16.288},
    
    # Common aliases
    'local': {'gpu': 'Local', 'gpu_count': 1, 'cost_per_hour': 0.0},
    'unknown': {'gpu': 'Unknown', 'gpu_count': 1, 'cost_per_hour': 0.0},
}

# GCP pricing (us-central1, on-demand)
GCP_PRICING = {
    'n1-standard-8-v100': {'gpu': 'V100', 'gpu_count': 1, 'cost_per_hour': 3.10},
    'n1-standard-16-v100-x4': {'gpu': '4x V100', 'gpu_count': 4, 'cost_per_hour': 12.40},
    
    'a2-highgpu-1g': {'gpu': 'A100', 'gpu_count': 1, 'cost_per_hour': 3.67},
    'a2-highgpu-8g': {'gpu': '8x A100', 'gpu_count': 8, 'cost_per_hour': 29.39},
    
    'n1-standard-4-t4': {'gpu': 'T4', 'gpu_count': 1, 'cost_per_hour': 0.51},
    'n1-standard-8-t4-x4': {'gpu': '4x T4', 'gpu_count': 4, 'cost_per_hour': 2.04},
}


@dataclass
class CostBreakdown:
    """Detailed cost breakdown"""
    instance_type: str
    gpu_type: str
    gpu_count: int
    hourly_rate: float
    duration_hours: float
    total_cost: float
    cost_per_epoch: Optional[float] = None
    cost_per_sample: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'instance_type': self.instance_type,
            'gpu_type': self.gpu_type,
            'gpu_count': self.gpu_count,
            'hourly_rate': round(self.hourly_rate, 3),
            'duration_hours': round(self.duration_hours, 2),
            'total_cost': round(self.total_cost, 2),
            'cost_per_epoch': round(self.cost_per_epoch, 2) if self.cost_per_epoch else None,
            'cost_per_sample': round(self.cost_per_sample, 6) if self.cost_per_sample else None,
        }


@dataclass
class OptimizationSavings:
    """Potential savings from optimization"""
    scenario: str
    description: str
    speedup_factor: float
    new_duration_hours: float
    new_cost: float
    savings: float
    savings_percent: int
    
    def to_dict(self) -> Dict:
        return {
            'scenario': self.scenario,
            'description': self.description,
            'speedup_factor': round(self.speedup_factor, 2),
            'new_duration_hours': round(self.new_duration_hours, 2),
            'new_cost': round(self.new_cost, 2),
            'savings': round(self.savings, 2),
            'savings_percent': self.savings_percent
        }


class CostEstimator:
    """
    Estimates training costs and calculates optimization savings
    """
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run = Run.query.get(run_id)
        if not self.run:
            raise ValueError(f"Run {run_id} not found")
        
        self.pricing = self._get_pricing()
    
    def _get_pricing(self) -> Dict:
        """Get pricing info for instance type"""
        instance_type = self.run.instance_type or 'unknown'
        
        # Try AWS pricing first
        if instance_type in AWS_PRICING:
            return {'provider': 'AWS', **AWS_PRICING[instance_type]}
        
        # Try GCP pricing
        if instance_type in GCP_PRICING:
            return {'provider': 'GCP', **GCP_PRICING[instance_type]}
        
        # Unknown instance
        return {
            'provider': 'Unknown',
            'gpu': 'Unknown',
            'gpu_count': 1,
            'cost_per_hour': 0.0
        }
    
    def calculate_cost(self) -> CostBreakdown:
        """Calculate actual training cost"""
        # Calculate duration
        if not self.run.started_at:
            raise ValueError("Run has no start time")
        
        end_time = self.run.ended_at or datetime.utcnow()
        duration_seconds = (end_time - self.run.started_at).total_seconds()
        duration_hours = duration_seconds / 3600
        
        # Calculate cost
        hourly_rate = self.pricing['cost_per_hour']
        total_cost = duration_hours * hourly_rate
        
        # Cost per epoch (if available)
        cost_per_epoch = None
        # TODO: Extract epoch count from metrics
        
        # Cost per sample (if throughput available)
        cost_per_sample = None
        if self.run.avg_throughput and self.run.avg_throughput > 0:
            samples_per_hour = self.run.avg_throughput * 3600
            if samples_per_hour > 0:
                cost_per_sample = hourly_rate / samples_per_hour
        
        return CostBreakdown(
            instance_type=self.run.instance_type or 'unknown',
            gpu_type=self.pricing['gpu'],
            gpu_count=self.pricing['gpu_count'],
            hourly_rate=hourly_rate,
            duration_hours=duration_hours,
            total_cost=total_cost,
            cost_per_epoch=cost_per_epoch,
            cost_per_sample=cost_per_sample
        )
    
    def estimate_optimization_savings(
        self,
        speedup_factor: float,
        scenario: str = "optimization"
    ) -> OptimizationSavings:
        """
        Calculate savings from a given speedup
        
        Args:
            speedup_factor: Expected speedup (e.g., 2.0 for 2x faster)
            scenario: Description of the optimization
        """
        cost_breakdown = self.calculate_cost()
        
        # Calculate new duration and cost
        new_duration = cost_breakdown.duration_hours / speedup_factor
        new_cost = new_duration * cost_breakdown.hourly_rate
        
        # Calculate savings
        savings = cost_breakdown.total_cost - new_cost
        savings_percent = int((savings / cost_breakdown.total_cost) * 100)
        
        return OptimizationSavings(
            scenario=scenario,
            description=f"With {speedup_factor}x speedup",
            speedup_factor=speedup_factor,
            new_duration_hours=new_duration,
            new_cost=new_cost,
            savings=savings,
            savings_percent=savings_percent
        )
    
    def suggest_instance_downsize(self, target_gpu_util: float = 85) -> Optional[Dict]:
        """
        Suggest cheaper instance based on actual GPU utilization
        
        Args:
            target_gpu_util: Target GPU utilization percentage
        """
        if not self.run.avg_gpu_util or self.run.avg_gpu_util >= target_gpu_util:
            return None
        
        # Calculate how much cheaper instance we could use
        util_ratio = self.run.avg_gpu_util / target_gpu_util
        
        # Find cheaper alternatives (simplified)
        current_hourly = self.pricing['cost_per_hour']
        target_hourly = current_hourly * util_ratio
        
        # Look for cheaper instances with similar GPU type
        current_gpu = self.pricing['gpu']
        alternatives = []
        
        pricing_db = {**AWS_PRICING, **GCP_PRICING}
        for instance, info in pricing_db.items():
            if (info['cost_per_hour'] < current_hourly and
                info['cost_per_hour'] >= target_hourly * 0.8 and
                current_gpu in info['gpu']):
                alternatives.append({
                    'instance_type': instance,
                    'hourly_rate': info['cost_per_hour'],
                    'gpu': info['gpu'],
                    'savings_per_hour': current_hourly - info['cost_per_hour']
                })
        
        if not alternatives:
            return None
        
        # Return best alternative
        best = min(alternatives, key=lambda x: abs(x['hourly_rate'] - target_hourly))
        
        cost_breakdown = self.calculate_cost()
        potential_savings = (current_hourly - best['hourly_rate']) * cost_breakdown.duration_hours
        
        return {
            'current_instance': self.run.instance_type,
            'current_cost': current_hourly,
            'current_gpu_util': round(self.run.avg_gpu_util, 1),
            'suggested_instance': best['instance_type'],
            'suggested_cost': best['hourly_rate'],
            'savings_per_hour': round(best['savings_per_hour'], 2),
            'estimated_savings': round(potential_savings, 2),
            'note': 'First fix data loading bottleneck before downsizing'
        }
    
    def generate_cost_report(self) -> Dict:
        """Generate comprehensive cost report"""
        cost_breakdown = self.calculate_cost()
        
        # Calculate optimization scenarios
        scenarios = []
        
        # Scenario 1: Fix I/O bottleneck (2x speedup)
        if self.run.avg_gpu_util and self.run.avg_gpu_util < 70:
            scenarios.append(
                self.estimate_optimization_savings(
                    speedup_factor=2.0,
                    scenario="Fix I/O bottleneck (add num_workers)"
                )
            )
        
        # Scenario 2: Moderate optimization (1.5x speedup)
        scenarios.append(
            self.estimate_optimization_savings(
                speedup_factor=1.5,
                scenario="General optimization"
            )
        )
        
        # Scenario 3: Instance right-sizing
        downsize_suggestion = self.suggest_instance_downsize()
        
        return {
            'run_id': self.run_id,
            'run_name': self.run.name,
            'cost_breakdown': cost_breakdown.to_dict(),
            'optimization_scenarios': [s.to_dict() for s in scenarios],
            'instance_recommendation': downsize_suggestion,
            'summary': {
                'total_cost': round(cost_breakdown.total_cost, 2),
                'max_potential_savings': round(scenarios[0].savings, 2) if scenarios else 0,
                'hourly_rate': round(cost_breakdown.hourly_rate, 2)
            }
        }
