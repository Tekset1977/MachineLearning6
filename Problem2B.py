import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# FORCE CPU USAGE (No GPU)
device = torch.device('cpu')
print(f'Using device: {device} (GPU credits exhausted - using CPU)')
print('NOTE: Training will be slower on CPU')

# ==========================================
# STEP 1: Load and Preprocess CIFAR-10 Data
# ==========================================

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("\nLoading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
# CPU: Smaller batch size for stability
trainloader = DataLoader(trainset, batch_size=128, shuffle=True,
                         num_workers=0)  # num_workers=0 for CPU

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False,
                        num_workers=0)  # num_workers=0 for CPU

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")

# ==========================================
# STEP 2: Define Residual Block
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, dropout_p=0.3):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Optional dropout
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

# ==========================================
# STEP 3: Define Three ResNet-10 Variants
# ==========================================

class ResNet10_WeightDecay(nn.Module):
    """ResNet-10 with Batch Normalization (Weight Decay in optimizer)"""
    def __init__(self, num_classes=10):
        super(ResNet10_WeightDecay, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=3, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_dropout=False))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_dropout=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet10_Dropout(nn.Module):
    """ResNet-10 with Batch Normalization + Dropout (p=0.3)"""
    def __init__(self, num_classes=10, dropout_p=0.3):
        super(ResNet10_Dropout, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, dropout_p=dropout_p)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout_p=dropout_p)
        self.layer3 = self._make_layer(128, 256, num_blocks=3, stride=2, dropout_p=dropout_p)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2, dropout_p=dropout_p)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_p):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_dropout=True, dropout_p=dropout_p))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_dropout=True, dropout_p=dropout_p))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet10_BatchNorm(nn.Module):
    """ResNet-10 with ONLY Batch Normalization (no additional regularization)"""
    def __init__(self, num_classes=10):
        super(ResNet10_BatchNorm, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=3, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_dropout=False))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_dropout=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ==========================================
# STEP 4: Training and Evaluation Functions
# ==========================================

def train_epoch(model, trainloader, criterion, optimizer, device, scaler=None):
    import sys
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs = inputs.to(device)  # Simple CPU transfer
        labels = labels.to(device)

        optimizer.zero_grad()

        # No mixed precision on CPU
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)  # Simple CPU transfer
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total

    return test_loss, test_acc

def train_model(model, model_name, trainloader, testloader, device, num_epochs=150, weight_decay=0.0):
    """Train a model and return results"""
    import sys

    print("\n" + "="*70)
    print(f"Training: {model_name}")
    print("="*70)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scaler = None  # No mixed precision on CPU

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            test_loss, test_acc = evaluate(model, testloader, criterion, device)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.2f}%')
            sys.stdout.flush()

        # Progress update every 25 epochs
        if (epoch + 1) % 25 == 0 and (epoch + 1) % 10 != 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min')
            sys.stdout.flush()

    training_time = time.time() - start_time
    final_test_loss, final_test_acc = evaluate(model, testloader, criterion, device)

    print(f"\n{model_name} - Training Complete!")
    print(f"Training Time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"Final Test Loss: {final_test_loss:.4f}")

    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'training_time': training_time,
        'final_test_acc': final_test_acc,
        'final_test_loss': final_test_loss,
        'final_train_acc': train_accuracies[-1],
        'final_train_loss': train_losses[-1]
    }

# ==========================================
# STEP 5: Train All Three Models
# ==========================================

results = {}

# Model 1: Weight Decay (lambda = 0.001)
print("\n" + "="*70)
print("EXPERIMENT 1: Weight Decay (lambda = 0.001)")
print("="*70)
model1 = ResNet10_WeightDecay(num_classes=10).to(device)
# No GPU optimizations on CPU
print(f"Model 1 parameters: {sum(p.numel() for p in model1.parameters()):,}")

results['weight_decay'] = train_model(model1, "ResNet-10 + Weight Decay (λ=0.001)",
                                      trainloader, testloader, device,
                                      num_epochs=150, weight_decay=0.001)

# Model 2: Dropout (p = 0.3)
print("\n" + "="*70)
print("EXPERIMENT 2: Dropout (p = 0.3)")
print("="*70)
model2 = ResNet10_Dropout(num_classes=10, dropout_p=0.3).to(device)
print(f"Model 2 parameters: {sum(p.numel() for p in model2.parameters()):,}")

results['dropout'] = train_model(model2, "ResNet-10 + Dropout (p=0.3)",
                                 trainloader, testloader, device,
                                 num_epochs=150, weight_decay=0.0)

# Model 3: Batch Normalization Only
print("\n" + "="*70)
print("EXPERIMENT 3: Batch Normalization Only (Baseline)")
print("="*70)
model3 = ResNet10_BatchNorm(num_classes=10).to(device)
print(f"Model 3 parameters: {sum(p.numel() for p in model3.parameters()):,}")

results['batch_norm'] = train_model(model3, "ResNet-10 + Batch Normalization",
                                    trainloader, testloader, device,
                                    num_epochs=150, weight_decay=0.0)

# ==========================================
# STEP 6: Compare Results
# ==========================================

print("\n" + "="*70)
print("FINAL COMPARISON - Part 2.b")
print("="*70)

comparison_data = []
for key in ['weight_decay', 'dropout', 'batch_norm']:
    r = results[key]
    comparison_data.append({
        'Method': r['model_name'],
        'Train Time (min)': f"{r['training_time']/60:.2f}",
        'Final Train Loss': f"{r['final_train_loss']:.4f}",
        'Final Train Acc': f"{r['final_train_acc']:.2f}%",
        'Final Test Loss': f"{r['final_test_loss']:.4f}",
        'Final Test Acc': f"{r['final_test_acc']:.2f}%",
        'Train-Test Gap': f"{r['final_train_acc'] - r['final_test_acc']:.2f}%"
    })

print("\n{:<45} {:<15} {:<18} {:<18} {:<18} {:<18} {:<15}".format(
    "Method", "Train Time", "Train Loss", "Train Acc", "Test Loss", "Test Acc", "Overfit Gap"))
print("-" * 150)
for data in comparison_data:
    print("{:<45} {:<15} {:<18} {:<18} {:<18} {:<18} {:<15}".format(
        data['Method'], data['Train Time (min)'], data['Final Train Loss'],
        data['Final Train Acc'], data['Final Test Loss'], data['Final Test Acc'],
        data['Train-Test Gap']))

# ==========================================
# STEP 7: Plot Comparison
# ==========================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

colors = {'weight_decay': 'blue', 'dropout': 'red', 'batch_norm': 'green'}
labels = {
    'weight_decay': 'Weight Decay (λ=0.001)',
    'dropout': 'Dropout (p=0.3)',
    'batch_norm': 'Batch Norm Only'
}

# Plot 1: Training Loss
for key in ['weight_decay', 'dropout', 'batch_norm']:
    ax1.plot(results[key]['train_losses'], label=labels[key],
             color=colors[key], linewidth=2, alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training Accuracy
for key in ['weight_decay', 'dropout', 'batch_norm']:
    ax2.plot(results[key]['train_accuracies'], label=labels[key],
             color=colors[key], linewidth=2, alpha=0.8)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
ax2.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Validation Loss
for key in ['weight_decay', 'dropout', 'batch_norm']:
    test_epochs = [i for i in range(150) if (i+1) % 10 == 0 or i == 0 or i == 149]
    ax3.plot(test_epochs[:len(results[key]['test_losses'])], results[key]['test_losses'],
             label=labels[key], color=colors[key], linewidth=2, marker='o', markersize=4, alpha=0.8)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Validation Loss', fontsize=12)
ax3.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Validation Accuracy
for key in ['weight_decay', 'dropout', 'batch_norm']:
    test_epochs = [i for i in range(150) if (i+1) % 10 == 0 or i == 0 or i == 149]
    ax4.plot(test_epochs[:len(results[key]['test_accuracies'])], results[key]['test_accuracies'],
             label=labels[key], color=colors[key], linewidth=2, marker='o', markersize=4, alpha=0.8)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax4.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
