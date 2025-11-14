"""
Simple MNIST training example with TrainOps monitoring
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# Add SDK to path (for development)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sdk'))

from trainops import TrainOpsMonitor


# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, monitor):
    """Training loop for one epoch"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Log metrics to TrainOps
        monitor.log_step(loss=loss.item())
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')


def test(model, device, test_loader):
    """Evaluation loop"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


def main():
    # Configuration
    batch_size = 64
    epochs = 5
    lr = 0.01
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Initialize TrainOps Monitor
    monitor = TrainOpsMonitor(
        run_name="mnist_simple_baseline",
        project="mnist_classification",
        instance_type="local",
        tags={"model": "SimpleCNN", "batch_size": str(batch_size)}
    )
    
    # Training with monitoring
    print("\n" + "="*50)
    print("Starting training with TrainOps monitoring")
    print(f"Run ID: {monitor.run_id}")
    print("="*50 + "\n")
    
    monitor.start_collection()
    
    try:
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, monitor)
            test_loss, accuracy = test(model, device, test_loader)
            
            # Log epoch metrics
            monitor.log_step(test_loss=test_loss, test_accuracy=accuracy)
            monitor.log_epoch(epoch)
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"View results at: http://localhost:3000/runs/{monitor.run_id}")
        print("="*50 + "\n")
        
    finally:
        monitor.finish()


if __name__ == '__main__':
    main()
