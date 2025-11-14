"""
ResNet on CIFAR-10 with TrainOps monitoring
Demonstrates different bottleneck scenarios
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdk'))
from trainops import TrainOpsMonitor


def get_data_loaders(batch_size=128, num_workers=0, slow_io=False):
    """Get CIFAR-10 data loaders"""
    
    # Intentionally slow transforms for bottleneck demo
    if slow_io:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # Add expensive CPU transforms
            transforms.RandomErasing(p=0.5),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    testset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, monitor):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Log metrics
        monitor.log_step(
            loss=loss.item(),
            accuracy=100. * correct / total,
            batch_idx=batch_idx
        )
        
        if batch_idx % 50 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {train_loss/(batch_idx+1):.3f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return train_loss / len(train_loader), 100. * correct / total


def test(model, device, test_loader):
    """Evaluate model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'  Test Loss: {test_loss:.3f} | Test Acc: {accuracy:.2f}%')
    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='ResNet CIFAR-10 with TrainOps')
    parser.add_argument('--scenario', type=str, default='baseline',
                        choices=['baseline', 'io_bound', 'optimized'],
                        help='Training scenario')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()
    
    # Configuration based on scenario
    scenarios = {
        'baseline': {
            'num_workers': 0,
            'slow_io': False,
            'description': 'No data loading optimization (single-threaded)'
        },
        'io_bound': {
            'num_workers': 0,
            'slow_io': True,
            'description': 'Intentionally slow I/O to demonstrate bottleneck'
        },
        'optimized': {
            'num_workers': 4,
            'slow_io': False,
            'description': 'Optimized with multi-worker data loading'
        }
    }
    
    config = scenarios[args.scenario]
    
    print("\n" + "="*70)
    print(f"Scenario: {args.scenario}")
    print(f"Description: {config['description']}")
    print(f"Batch size: {args.batch_size}, Workers: {config['num_workers']}")
    print("="*70 + "\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Data
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=config['num_workers'],
        slow_io=config['slow_io']
    )
    
    # Model
    model = models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize TrainOps
    monitor = TrainOpsMonitor(
        run_name=f"resnet18_cifar10_{args.scenario}",
        project="cifar10_classification",
        instance_type="local",
        tags={
            "model": "ResNet18",
            "scenario": args.scenario,
            "batch_size": str(args.batch_size),
            "num_workers": str(config['num_workers'])
        }
    )
    
    print(f"Run ID: {monitor.run_id}")
    print("-"*70 + "\n")
    
    # Training
    monitor.start_collection()
    
    try:
        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            print(f'Epoch {epoch}/{args.epochs}')
            
            train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizer, criterion, monitor
            )
            test_loss, test_acc = test(model, device, test_loader)
            
            # Log epoch metrics
            monitor.log_step(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                lr=scheduler.get_last_lr()[0]
            )
            monitor.log_epoch(epoch)
            
            scheduler.step()
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            print()
        
        print("="*70)
        print(f"Training complete! Best accuracy: {best_acc:.2f}%")
        print(f"View results: trainops runs show {monitor.run_id}")
        print(f"Dashboard: http://localhost:3000/runs/{monitor.run_id}")
        print("="*70 + "\n")
        
    finally:
        monitor.finish()


if __name__ == '__main__':
    main()
