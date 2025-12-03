import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


torch.manual_seed(42)
np.random.seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')


# Define transformations for training and testing
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
print("\nLoading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True,
                         num_workers=2, pin_memory=True, persistent_workers=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=256, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")

#Define ResNet Block with Skip Connections

class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection
    Implements: output = F(x) + x (skip connection)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, use 1x1 conv to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Save input for skip connection
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: Add input to output
        out += self.shortcut(identity)
        out = self.relu(out)

        return out

#Define ResNet-10 Architecture

class ResNet10(nn.Module):
    """
    ResNet-10: ResNet with 10 residual blocks
    Architecture similar to lecture with skip connections
    """
    def __init__(self, num_classes=10):
        super(ResNet10, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet blocks (10 blocks total)
        # Layer 1: 2 blocks with 64 channels
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # Layer 2: 2 blocks with 128 channels (downsample)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # Layer 3: 3 blocks with 256 channels (downsample)
        self.layer3 = self._make_layer(128, 256, num_blocks=3, stride=2)

        # Layer 4: 3 blocks with 512 channels (downsample)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []

        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet blocks with skip connections
        x = self.layer1(x)  # 2 blocks
        x = self.layer2(x)  # 2 blocks
        x = self.layer3(x)  # 3 blocks
        x = self.layer4(x)  # 3 blocks
        # Total: 2 + 2 + 3 + 3 = 10 blocks

        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc(x)

        return x

# Initialize the model
model = ResNet10(num_classes=10).to(device)
print("\n" + "="*60)
print("RESNET-10 ARCHITECTURE (10 Residual Blocks)")
print("="*60)
print(model)

#Calculate Model Size

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32 (4 bytes)

print(f"\n{'MODEL STATISTICS (Part 2.a - ResNet-10)':^60}")
print("="*60)
print(f"Total Residual Blocks: 10 (2+2+3+3)")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {model_size_mb:.2f} MB")
print("="*60)

#Define Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use automatic mixed precision for faster training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
print(f"Using mixed precision training: {scaler is not None}")

#Training Function

def train_epoch(model, trainloader, criterion, optimizer, device, epoch, num_epochs, scaler=None):
    """Train for one epoch with progress printing"""
    import sys
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    # Print completion
    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')
    sys.stdout.flush()

    return epoch_loss, epoch_acc

#Evaluation Function

def evaluate(model, testloader, criterion, device):
    """Evaluate the model on test set"""
    import sys
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total

    return test_loss, test_acc

#Train for 300 Epochs

print("\n" + "="*60)
print("Starting Training for 300 Epochs (Part 2.a - ResNet-10)")
print("="*60 + "\n")
import sys
sys.stdout.flush()

num_epochs = 300
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Start timing
start_time = time.time()

for epoch in range(num_epochs):
    # Train with progress printing
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, epoch, num_epochs, scaler)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Evaluate on test set every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Print test results
        print(f'         Val Loss: {test_loss:.4f} - Val Accuracy: {test_acc:.4f}')
        sys.stdout.flush()

    # Print time estimate every 10 epochs
    if (epoch + 1) % 10 == 0:
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining = avg_time_per_epoch * remaining_epochs
        print(f'\n  >>> Elapsed: {elapsed_time/60:.1f}min | Est. Remaining: {estimated_remaining/60:.1f}min\n')
        sys.stdout.flush()

# End timing
end_time = time.time()
training_time = end_time - start_time

#Final Evaluation

print("\n" + "="*60)
print("Training Complete! (Part 2.a - ResNet-10)")
print("="*60)

final_test_loss, final_test_acc = evaluate(model, testloader, criterion, device)

print(f"\n{'FINAL RESULTS - Part 2.a (ResNet-10)':^60}")
print("="*60)
print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}%")
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"Total Parameters: {total_params:,}")
print(f"Residual Blocks: 10")
print("="*60)

#Overfitting Analysis

print(f"\n{'OVERFITTING ANALYSIS':^60}")
print("="*60)

# Calculate train-test gap
train_test_acc_gap = train_accuracies[-1] - final_test_acc
train_test_loss_gap = final_test_loss - train_losses[-1]

print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
print(f"Accuracy Gap (Train - Test): {train_test_acc_gap:.2f}%")
print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Loss Gap (Test - Train): {train_test_loss_gap:.4f}")

# Overfitting assessment
print(f"\n{'Overfitting Assessment:':^60}")
if train_test_acc_gap > 10:
    print("⚠️  SIGNIFICANT OVERFITTING DETECTED")
    print(f"   The model performs {train_test_acc_gap:.2f}% better on training data")
elif train_test_acc_gap > 5:
    print("⚠️  MODERATE OVERFITTING")
    print(f"   The model performs {train_test_acc_gap:.2f}% better on training data")
else:
    print("✓  Good generalization - minimal overfitting")
    print(f"   Training-test gap is only {train_test_acc_gap:.2f}%")
print("="*60)

#Comparison with Problem 1.b

print(f"\n{'COMPARISON: Part 2.a (ResNet-10) vs Part 1.b (Deeper CNN)':^60}")
print("="*60)
print("\nDeeper CNN (Part 1.b - 4 Conv Layers):")
print("  - Training Time: [Record from 1.b] minutes")
print("  - Test Accuracy: [Record from 1.b]%")
print("  - Model Parameters: [Record from 1.b]")
print("  - Model Size: [Record from 1.b] MB")
print("  - Overfitting Gap: [Record from 1.b]%")
print("\nResNet-10 (Part 2.a - 10 Residual Blocks):")
print(f"  - Training Time: {training_time/60:.2f} minutes")
print(f"  - Test Accuracy: {final_test_acc:.2f}%")
print(f"  - Model Parameters: {total_params:,}")
print(f"  - Model Size: {model_size_mb:.2f} MB")
print(f"  - Overfitting Gap: {train_test_acc_gap:.2f}%")
print("\nExpected Observations:")
print("  - ResNet should have SIGNIFICANTLY MORE parameters")
print("  - ResNet should achieve HIGHER test accuracy")
print("  - ResNet should show LESS overfitting (skip connections help)")
print("  - Training time may be longer due to more blocks")
print("="*60)

#Plot Training History

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss
ax1.plot(train_losses, label='Training Loss', linewidth=2)
if test_losses:
    test_epochs = [i for i in range(num_epochs) if (i+1) % 5 == 0 or i == 0 or i == num_epochs-1]
    ax1.plot(test_epochs[:len(test_losses)], test_losses, label='Validation Loss',
             linewidth=2, marker='o', markersize=3)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss - ResNet-10 (Part 2.a)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Accuracy
ax2.plot(train_accuracies, label='Training Accuracy', linewidth=2)
if test_accuracies:
    test_epochs = [i for i in range(num_epochs) if (i+1) % 5 == 0 or i == 0 or i == num_epochs-1]
    ax2.plot(test_epochs[:len(test_accuracies)], test_accuracies,
             label='Validation Accuracy', linewidth=2, marker='o', markersize=3)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Validation Accuracy - ResNet-10 (Part 2.a)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_resnet10.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining curves saved as 'training_history_resnet10.png'")

#Save the Model

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'test_accuracy': final_test_acc,
    'total_params': total_params,
}, 'cifar10_resnet10_model.pth')
