import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms, models
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

TRAIN_DIR = "../input/intel-image-classification/seg_train/seg_train/"
VALID_DIR = "../input/intel-image-classification/seg_test/seg_test/"

train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DIR)
valid_dataset = torchvision.datasets.ImageFolder(root=VALID_DIR)


IMAGE_SIZE = 128

gen = iter(train_dataset)

data_transforms = transforms.Compose(
    [transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]), transforms.ToTensor()]
)

train_dataset = torchvision.datasets.ImageFolder(
    root=TRAIN_DIR, transform=data_transforms
)
valid_dataset = torchvision.datasets.ImageFolder(
    root=VALID_DIR, transform=data_transforms
)
img, target = next(iter(train_dataset))
print(f"Image data type: {type(img)}")
print(f"     Image size: {img.shape}")


BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset,  # our raw data
    batch_size=BATCH_SIZE,  # the size of batches the dataloader returns
    shuffle=True,  # shuffle our data before batching
    drop_last=False,  # not dropping the last batch even if it's smaller than batch_size
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,  # our raw validation data
    batch_size=BATCH_SIZE,  # the size of batches the dataloader returns
    shuffle=True,
)


def trainer(
    model, criterion, optimizer, trainloader, validloader, epochs=5, verbose=True
):
    """Simple training wrapper for PyTorch network."""

    train_loss, valid_loss, valid_accuracy, train_accuracy = [], [], [], []
    for epoch in range(epochs):
        train_batch_loss = 0
        train_batch_acc = 0
        valid_batch_loss = 0
        valid_batch_acc = 0

        # Training
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # Zero all the gradients w.r.t. parameters
            y_hat = model(X)  # initialize model
            _, y_hat_labels = torch.softmax(y_hat, dim=1).topk(
                1, dim=1
            )  # Assigning class label to prediction
            loss = criterion(y_hat, y)  # Calculate loss based on output
            loss.backward()  # Calculate gradients w.r.t. parameters
            optimizer.step()  # Update parameters
            train_batch_loss += loss.item()  # Add loss for this batch to running total
            train_batch_acc += (
                (y_hat_labels.squeeze() == y).type(torch.float32).mean().item()
            )
        train_loss.append(train_batch_loss / len(trainloader))
        train_accuracy.append(
            train_batch_acc / len(trainloader)
        )  # accuracy of the train set

        # Validation
        model.eval()
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
            for X, y in validloader:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                _, y_hat_labels = torch.softmax(y_hat, dim=1).topk(1, dim=1)
                loss = criterion(y_hat, y)
                valid_batch_loss += loss.item()
                valid_batch_acc += (
                    (y_hat_labels.squeeze() == y).type(torch.float32).mean().item()
                )
        valid_loss.append(valid_batch_loss / len(validloader))
        valid_accuracy.append(
            valid_batch_acc / len(validloader)
        )  # accuracy of the valid set

        model.train()

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch + 1}:",
                f"Train Loss: {train_loss[-1]:.3f}.",
                f"Valid Loss: {valid_loss[-1]:.3f}.",
                f"Train Accuracy: {train_accuracy[-1]:.2f}.",
                f"Valid Accuracy: {valid_accuracy[-1]:.2f}.",
            )

    results = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
    }
    return results


densenet = models.densenet121(pretrained=True)

for param in densenet.parameters():  # Freeze parameters so we don't update them
    param.requires_grad = False

new_layers = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 6))
densenet.classifier = new_layers

torch.manual_seed(2018)
densenet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet.parameters(), lr=0.0005)

results = trainer(densenet, criterion, optimizer, train_loader, valid_loader, epochs=20)
