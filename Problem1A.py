import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

#GPU Check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')



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
print("Loading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
# FASTER: Larger batch size and pin_memory for GPU
trainloader = DataLoader(trainset, batch_size=256, shuffle=True,
                         num_workers=2, pin_memory=True, persistent_workers=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
# FASTER: Larger batch size for testing
testloader = DataLoader(testset, batch_size=256, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")
#CNN Architecture

class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Activation function
        self.relu = nn.ReLU()

        # Fully connected layers
        # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        # So: 128 channels * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
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

        # Flatten
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

#Init
model = CIFAR10_CNN(num_classes=10).to(device)
print("\nModel Architecture:")
print(model)

# Model Size


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32 

print(f"\nModel Statistics:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {model_size_mb:.2f} MB")

# Define Loss Function and Optimiser

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use automatic mixed precision for faster training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
print(f"Using mixed precision training: {scaler is not None}")

#Training 
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

    # Print completion (this will show EVERY epoch like TensorFlow)
    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')
    sys.stdout.flush()  # Force print to show immediately

    return epoch_loss, epoch_acc

#Eval

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

#training 300 epkj

print("\n" + "="*50)
print("Starting Training for 300 Epochs")
print("="*50 + "\n")
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

    # Evaluate on test set every 5 epochs (not every epoch to save time)
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Print test results (like your TensorFlow code with val_loss and val_accuracy)
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

#print

print("\n" + "="*50)
print("Training Complete!")
print("="*50)

final_test_loss, final_test_acc = evaluate(model, testloader, criterion, device)

print(f"\n{'FINAL RESULTS':^50}")
print("="*50)
print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"Total Parameters: {total_params:,}")
print("="*50)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Loss
ax1.plot(train_losses, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss over Epochs')
ax1.legend()
ax1.grid(True)

# Plot Accuracy
ax2.plot(train_accuracies, label='Training Accuracy')
if test_accuracies:
    test_epochs = list(range(0, num_epochs, 10))
    if 0 not in test_epochs:
        test_epochs = [0] + test_epochs
    ax2.plot(test_epochs[:len(test_accuracies)], test_accuracies,
             label='Test Accuracy', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy over Epochs')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining curves saved as 'training_history.png'")


torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'test_accuracy': final_test_acc,
}, 'cifar10_cnn_model.pth')
