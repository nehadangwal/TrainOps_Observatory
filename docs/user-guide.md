# TrainOps Observatory - User Guide

Complete guide to using TrainOps for monitoring ML training workflows.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Instrumenting Your Training Code](#instrumenting-your-training-code)
4. [Using the CLI](#using-the-cli)
5. [Understanding Metrics](#understanding-metrics)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (for backend)
- PyTorch or TensorFlow (for training)
- NVIDIA GPU (optional, for GPU metrics)

### Step 1: Start the Backend

```bash
# Clone repository
git clone https://github.com/nehadangwal/trainops-observatory
cd trainops-observatory

# Start services
docker-compose up -d

# Verify it's running
curl http://localhost:5000/health
```

### Step 2: Install SDK

```bash
cd sdk
pip install -e .

# Verify installation
trainops --help
```

---

## Quick Start

### Basic Example

```python
from trainops import TrainOpsMonitor
import torch

# Create monitor
monitor = TrainOpsMonitor(
    run_name="my_first_run",
    project="experiments"
)

# Use decorator for automatic tracking
@monitor.track_training
def train_epoch(model, dataloader, optimizer):
    for batch in dataloader:
        # ... your training code ...
        loss = compute_loss(...)
        
        # Log metrics
        monitor.log_step(loss=loss.item())

# Train
for epoch in range(10):
    train_epoch(model, train_loader, optimizer)
    monitor.log_epoch(epoch)

# Finish
monitor.finish()
```

### View Results

```bash
# List runs
trainops runs list

# Show details
trainops runs show <run-id>

# View metrics
trainops runs metrics <run-id>
```

---

## Instrumenting Your Training Code

### Method 1: Decorator (Recommended)

```python
from trainops import TrainOpsMonitor

monitor = TrainOpsMonitor(run_name="experiment_1", project="mnist")

@monitor.track_training
def train():
    for epoch in range(epochs):
        for batch in dataloader:
            # ... training code ...
            monitor.log_step(loss=loss.item(), accuracy=acc)
        monitor.log_epoch(epoch)

train()
monitor.finish()
```

**Pros:**
- Automatic start/stop of metric collection
- Clean, minimal code changes
- Automatic cleanup on errors

### Method 2: Context Manager

```python
from trainops import TrainOpsMonitor

with TrainOpsMonitor(run_name="experiment_2", project="mnist") as monitor:
    for epoch in range(epochs):
        for batch in dataloader:
            # ... training code ...
            monitor.log_step(loss=loss.item())
        monitor.log_epoch(epoch)
```

**Pros:**
- Automatic cleanup with `with` statement
- Good for single training session

### Method 3: Manual Control

```python
from trainops import TrainOpsMonitor

monitor = TrainOpsMonitor(run_name="experiment_3", project="mnist")
monitor.start_collection()

try:
    for epoch in range(epochs):
        for batch in dataloader:
            # ... training code ...
            monitor.log_step(loss=loss.item())
        monitor.log_epoch(epoch)
finally:
    monitor.finish()
```

**Pros:**
- Maximum control
- Good for complex workflows

---

## Logging Metrics

### Step-Level Metrics

Log metrics after each training step:

```python
monitor.log_step(
    loss=loss.item(),
    accuracy=accuracy,
    learning_rate=lr,
    custom_metric=value
)
```

All metrics are stored in the `custom` field and can be queried later.

### Epoch-Level Markers

Mark epoch boundaries:

```python
monitor.log_epoch(epoch_num)
```

This helps with time-series analysis and visualization.

---

## Using the CLI

### List Runs

```bash
# All runs
trainops runs list

# Filter by project
trainops runs list --project mnist

# Filter by status
trainops runs list --status running

# Limit results
trainops runs list --limit 10
```

### Show Run Details

```bash
trainops runs show <run-id>
```

Output:
```
======================================================================
Run: resnet50_baseline
======================================================================

ID:              a1b2c3d4-...
Project:         image_classification
Status:          completed
Instance Type:   p3.2xlarge
Started:         2024-11-13 10:30:00
Ended:           2024-11-13 11:45:00
Duration:        1h 15m 0s

----------------------------------------------------------------------
Performance Summary
----------------------------------------------------------------------
Avg GPU Util:    75.3%
Avg CPU Util:    42.1%
Avg Throughput:  128.5 samples/sec
Max GPU Memory:  89.2%
Metrics Count:   450

======================================================================
View metrics: trainops runs metrics a1b2c3d4-...
```

### View Metrics

```bash
# Show recent metrics
trainops runs metrics <run-id>

# Show last N metrics
trainops runs metrics <run-id> --tail --limit 20

# Show first N metrics
trainops runs metrics <run-id> --limit 20
```

### List Projects

```bash
trainops projects
```

Output:
```
Project                  Runs
---------------------  ------
mnist                      12
cifar10                     8
imagenet                    3
```

### Platform Statistics

```bash
trainops stats
```

Output:
```
==================================================
TrainOps Observatory - Statistics
==================================================

Total Runs:      23
Running:         2
Completed:       21
Total Metrics:   45,632

==================================================
```

### Delete Runs

```bash
# With confirmation
trainops runs delete <run-id>

# Skip confirmation
trainops runs delete <run-id> --yes
```

---

## Understanding Metrics

### System Metrics (Collected Automatically)

| Metric | Description | Unit |
|--------|-------------|------|
| `gpu_util` | GPU utilization | % |
| `gpu_memory_used` | GPU memory used | GB |
| `gpu_memory_util` | GPU memory utilization | % |
| `cpu_util` | CPU utilization | % |
| `system_memory` | System RAM usage | % |
| `io_read_mb` | Disk read | MB |
| `io_write_mb` | Disk write | MB |
| `throughput` | Training throughput | samples/sec |

### Custom Metrics (User-Logged)

Any metrics you log via `monitor.log_step()` are stored in the `custom` field:

```python
monitor.log_step(
    loss=0.5,
    accuracy=0.95,
    learning_rate=0.001,
    my_custom_metric=42
)
```

---

## Configuration

### Environment Variables

Configure TrainOps behavior via environment variables:

```bash
# API Configuration
export TRAINOPS_API_URL="http://localhost:5000"
export TRAINOPS_API_TIMEOUT=10

# Collection Intervals
export TRAINOPS_COLLECT_INTERVAL=10  # seconds
export TRAINOPS_SEND_INTERVAL=30     # seconds

# Logging
export TRAINOPS_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Features
export TRAINOPS_ENABLE_GPU=true
export TRAINOPS_FAIL_ON_ERROR=false  # Continue training even if API fails
```

### Programmatic Configuration

```python
from trainops import TrainOpsMonitor

monitor = TrainOpsMonitor(
    run_name="my_run",
    project="my_project",
    api_url="http://custom-server:5000",
    instance_type="p3.8xlarge",
    tags={"team": "research", "experiment": "baseline"},
    collect_interval=5,   # Collect every 5 seconds
    send_interval=15      # Send every 15 seconds
)
```

---

## Troubleshooting

### SDK Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'trainops'`

**Solution:**
```bash
cd sdk
pip install -e .
python -c "import trainops; print('✓ Installed')"
```

---

### Backend Connection Issues

**Problem:** `Error: Could not connect to TrainOps API`

**Solution:**
```bash
# Check if backend is running
docker-compose ps

# Check health
curl http://localhost:5000/health

# View logs
docker-compose logs backend

# Restart
docker-compose restart backend
```

---

### GPU Metrics Not Available

**Problem:** GPU metrics show as N/A

**Solutions:**

1. **Install pynvml:**
   ```bash
   pip install pynvml
   ```

2. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```

3. **Verify GPU access:**
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

---

### Metrics Not Appearing

**Problem:** Metrics not showing up in CLI/dashboard

**Checklist:**
1. ✓ Backend is running
2. ✓ Run was created successfully (check logs)
3. ✓ `monitor.log_step()` is being called
4. ✓ Sufficient time has passed (metrics sent every 30s)
5. ✓ No network errors in logs

**Debug:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from trainops import TrainOpsMonitor
# ... will show detailed logs
```

---

### High Overhead

**Problem:** Training is noticeably slower with TrainOps

**Solutions:**

1. **Increase collection interval:**
   ```python
   monitor = TrainOpsMonitor(
       ...,
       collect_interval=30,  # Collect less frequently
       send_interval=60
   )
   ```

2. **Disable GPU metrics if not needed:**
   ```bash
   export TRAINOPS_ENABLE_GPU=false
   ```

3. **Profile to verify overhead:**
   ```python
   import time
   start = time.time()
   # ... training code ...
   print(f"Duration: {time.time() - start:.2f}s")
   ```

Expected overhead: < 1% of training time

---

## Best Practices

### 1. Use Descriptive Run Names

```python
# Bad
monitor = TrainOpsMonitor(run_name="run1")

# Good
monitor = TrainOpsMonitor(
    run_name="resnet50_imagenet_lr0.1_batch256",
    tags={"model": "resnet50", "dataset": "imagenet"}
)
```

### 2. Log Key Metrics

```python
# Log training metrics
monitor.log_step(
    loss=loss.item(),
    accuracy=acc,
    learning_rate=scheduler.get_last_lr()[0]
)

# Log validation metrics at epoch end
monitor.log_step(
    val_loss=val_loss,
    val_accuracy=val_acc
)
```

### 3. Use Projects to Organize

```python
# Group related experiments
monitor = TrainOpsMonitor(
    run_name="experiment_v3",
    project="model_compression"  # All compression experiments together
)
```

### 4. Handle Errors Gracefully

```python
monitor = TrainOpsMonitor(...)

try:
    # Training code
    train()
except KeyboardInterrupt:
    print("Training interrupted")
finally:
    monitor.finish()  # Always cleanup
```

---

## Advanced Usage

### Custom Instance Detection

```python
import requests

def detect_aws_instance():
    """Detect AWS instance type"""
    try:
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/instance-type',
            timeout=1
        )
        return response.text
    except:
        return 'unknown'

monitor = TrainOpsMonitor(
    run_name="my_run",
    instance_type=detect_aws_instance()
)
```

### Multiple Runs in One Script

```python
# Hyperparameter sweep
for lr in [0.001, 0.01, 0.1]:
    monitor = TrainOpsMonitor(
        run_name=f"sweep_lr_{lr}",
        project="hyperparameter_search",
        tags={"lr": str(lr)}
    )
    
    with monitor:
        train(learning_rate=lr)
```

---

## Getting Help

- **Documentation:** [docs/](/)
- **GitHub Issues:** [github.com/nehadangwal/trainops-observatory/issues](https://github.com/nehadangwal/trainops-observatory/issues)
- **Examples:** [examples/](../examples/)

---

**Next:** [Technical Deep-Dive](technical.md) | [Back to README](../README.md)
