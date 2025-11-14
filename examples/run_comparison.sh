#!/bin/bash

# Run comparison experiments to demonstrate bottleneck detection
# This creates 3 runs with different characteristics

echo "========================================"
echo "TrainOps Observatory - Comparison Demo"
echo "========================================"
echo ""
echo "This will run 3 training experiments:"
echo "  1. Baseline (no optimization)"
echo "  2. I/O Bound (intentionally slow)"
echo "  3. Optimized (multi-worker loading)"
echo ""
echo "Each run takes ~5 minutes"
echo ""

read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Check if ResNet script exists
if [ ! -f "resnet_cifar10.py" ]; then
    echo "Error: resnet_cifar10.py not found"
    echo "Run this script from the examples/ directory"
    exit 1
fi

echo ""
echo "Run 1/3: Baseline scenario"
echo "----------------------------------------"
python resnet_cifar10.py --scenario baseline --epochs 3

echo ""
echo "Run 2/3: I/O Bound scenario"
echo "----------------------------------------"
python resnet_cifar10.py --scenario io_bound --epochs 3

echo ""
echo "Run 3/3: Optimized scenario"
echo "----------------------------------------"
python resnet_cifar10.py --scenario optimized --epochs 3

echo ""
echo "========================================"
echo "All runs complete!"
echo "========================================"
echo ""
echo "View comparison:"
echo "  trainops runs list --project cifar10_classification"
echo ""
echo "Expected results:"
echo "  • Baseline: Moderate GPU utilization (~60-70%)"
echo "  • I/O Bound: Low GPU utilization (~40-50%) due to slow data loading"
echo "  • Optimized: High GPU utilization (~80-90%) with parallel loading"
echo ""
