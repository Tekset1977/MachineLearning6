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

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')

# ==========================================
# STEP 1: Load and Preprocess CIFAR-10 Data
# ==========================================

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

#Define the DEEPER CNN Architecture (4 Conv Layers)

class CIFAR10_DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_DeepCNN, self).__init__()

        # Convolutional layers (NOW 4 LAYERS instead of 3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # NEW LAYER

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Activation function
        self.relu = nn.ReLU()

        # Fully connected layers
        # After 4 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        # So: 256 channels * 2 * 2 = 1024 (ADJUSTED for 4 conv layers)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16

        # Conv Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8

        # Conv Block 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4

        # Conv Block 4 (NEW)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)  # 4x4 -> 2x2

        # Flatten
        x = x.view(-1, 256 * 2 * 2)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Initialize the model
model = CIFAR10_DeepCNN(num_classes=10).to(device)
print("\n" + "="*60)
print("DEEPER MODEL ARCHITECTURE (4 Convolutional Layers)")
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

print(f"\n{'MODEL STATISTICS (Part 1.b - Deeper CNN)':^60}")
print("="*60)
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
print("Starting Training for 300 Epochs (Part 1.b - Deeper CNN)")
print("="*60 + "\n")
import sys
sys.stdout.flush()

num_epochs = 300
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

#Start timing
start_time = time.time()

for epoch in range(num_epochs):
    #Train with progress printing
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, epoch, num_epochs, scaler)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    #Evaluate on test set every 5 epochs (not every epoch to save time)
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Print test results
        print(f'         Val Loss: {test_loss:.4f} - Val Accuracy: {test_acc:.4f}')
        sys.stdout.flush()

    #Print time estimate every 10 epochs
    if (epoch + 1) % 10 == 0:
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining = avg_time_per_epoch * remaining_epochs
        print(f'\n  >>> Elapsed: {elapsed_time/60:.1f}min | Est. Remaining: {estimated_remaining/60:.1f}min\n')
        sys.stdout.flush()

#End timing
end_time = time.time()
training_time = end_time - start_time

#Final Evaluation

print("\n" + "="*60)
print("Training Complete! (Part 1.b - Deeper CNN)")
print("="*60)

final_test_loss, final_test_acc = evaluate(model, testloader, criterion, device)

print(f"\n{'FINAL RESULTS - Part 1.b (Deeper CNN)':^60}")
print("="*60)
print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}%")
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"Total Parameters: {total_params:,}")
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
    print("  SIGNIFICANT OVERFITTING DETECTED")
    print(f"   The model performs {train_test_acc_gap:.2f}% better on training data")
elif train_test_acc_gap > 5:
    print("  MODERATE OVERFITTING")
    print(f"   The model performs {train_test_acc_gap:.2f}% better on training data")
else:
    print("✓  Good generalization - minimal overfitting")
    print(f"   Training-test gap is only {train_test_acc_gap:.2f}%")
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
ax1.set_title('Training and Validation Loss - Deeper CNN (Part 1.b)', fontsize=14)
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
ax2.set_title('Training and Validation Accuracy - Deeper CNN (Part 1.b)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_deeper_cnn.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining curves saved as 'training_history_deeper_cnn.png'")

#Save the Model

torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'test_accuracy': final_test_acc,
    'total_params': total_params,
}, 'cifar10_deeper_cnn_model.pth')
