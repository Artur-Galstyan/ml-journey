import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

# Add multiprocessing safety for macOS
if __name__ == "__main__":
    # This fixes the multiprocessing issue on macOS
    multiprocessing.set_start_method("spawn", force=True)

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 200
batch_size = 128
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4

# CIFAR-10 specific mean and std for normalization
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]

# Data augmentation and normalization
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Fix for multiprocessing issue on macOS
    pin_memory=True if torch.cuda.is_available() else False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # Fix for multiprocessing issue on macOS
    pin_memory=True if torch.cuda.is_available() else False,
)

# Classes in CIFAR-10
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Create modified ResNet-18 model for CIFAR-10
def create_resnet18_cifar():
    model = resnet18(weights=None)

    # # Modify the first convolutional layer for CIFAR-10
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # # Remove the max pooling layer
    # model.maxpool = nn.Identity()

    # Adjust the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


# Initialize model
model = create_resnet18_cifar().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
)

# Learning rate scheduler (reduce on plateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.1, patience=10
)

# For storing metrics
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []


# Training function
def train_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Calculate metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print statistics every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    return train_loss, train_acc


# Evaluation function
def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100.0 * correct / total

    return test_loss, test_acc


# Main function to run everything
def main():
    global model, optimizer, scheduler, criterion, train_loader, test_loader

    print("Starting training...")
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()

        train_loss, train_acc = train_epoch()
        test_loss, test_acc = evaluate()

        # Update learning rate scheduler
        scheduler.step(test_acc)

        # Save metrics
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "resnet18_cifar10_best.pth")

        # Print epoch statistics
        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
            f"Best Acc: {best_acc:.2f}%"
        )

    # Plot training and validation curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(test_loss_history, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(test_acc_history, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("resnet18_cifar10_training.png")
    plt.show()

    print(f"Best Test Accuracy: {best_acc:.2f}%")


# Run the main function
if __name__ == "__main__":
    main()
