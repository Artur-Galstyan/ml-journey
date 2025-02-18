import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import tensorflow_datasets as tfds

(train_dataset, test_dataset), info = tfds.load(
    "cats_vs_dogs",
    split=("train[:80%]", "train[80%:]"),
    with_info=True,
    as_supervised=True,
)  # pyright: ignore


def preprocess_dataset(dataset):
    images = []
    labels = []
    for image, label in tqdm(dataset, desc="Preprocessing"):
        # Convert to float and normalize
        image = tf.cast(image, tf.float32) / 255.0  # pyright: ignore

        # Resize to fixed dimensions
        image = tf.image.resize(image, (224, 224))

        # Convert to numpy and transpose to channel-first
        image = image.numpy()  # pyright: ignore
        image = np.transpose(image, (2, 0, 1))

        images.append(image)
        labels.append(label)

    return np.stack(images), np.array(labels)


print("Preprocessing training dataset...")
train_images, train_labels = preprocess_dataset(train_dataset)
print("Preprocessing test dataset...")
test_images, test_labels = preprocess_dataset(test_dataset)

# Convert to PyTorch datasets
train_dataset = TensorDataset(
    torch.FloatTensor(train_images), torch.FloatTensor(train_labels).view(-1, 1)
)
test_dataset = TensorDataset(
    torch.FloatTensor(test_images), torch.FloatTensor(test_labels).view(-1, 1)
)

# Create data loaders
BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class LocalResponseNorm(nn.Module):
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        return nn.functional.local_response_norm(
            x, self.size, alpha=self.alpha, beta=self.beta, k=self.k
        )


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item() * len(target)

            # Compute accuracy
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target).sum().item()
            total += len(target)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and move to device
    model = AlexNet().to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    n_epochs = 10
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        eval_loss, accuracy = evaluate(model, test_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Eval loss: {eval_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
