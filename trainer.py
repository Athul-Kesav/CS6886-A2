import os
import random
import torch
import torchvision.models as models
import torch.nn as nn
import csv
import numpy as np
from load_dataset import train_loader, test_loader

SEED = 51
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # deterministic mode disables cuDNN's autotuner

csv_file = "training_metrics.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])


NUM_EPOCHS = 15
LR = 1e-3

def get_mobilenet_v2(num_classes=10):
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights, width_mult=1.0)

    first_conv = model.features[0][0]

    if isinstance(first_conv, nn.Conv2d):
        first_conv.stride = (1, 1)

    classifier = model.classifier[1]
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Expected MobileNet-v2 classifier[1] to be nn.Linear")

    in_features = classifier.in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

model = get_mobilenet_v2(num_classes=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # Decay LR by 5% every epoch
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.20)  # Reduce LR by 0.25 every 2 epochs

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_acc = correct / total

    scheduler.step()

    # Validation phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    epoch_test_loss = val_loss / val_total
    epoch_test_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Test Acc: {epoch_test_acc*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc])

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

top1_accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
print(f"Final Test Top-1 Accuracy: {top1_accuracy:.2f}%")

# Generate Confusion Matrix for failure mode discussion
# cm = confusion_matrix(all_labels, all_preds)
# print("Confusion Matrix:\n", cm)

np.savez("test_predictions.npz", preds=all_preds, labels=all_labels)
print("Training complete. Metrics saved to CSV.")

os.makedirs("./model", exist_ok=True)
torch.save(model.state_dict(), "./model/mobilenetv2_cifar10_baseline.pth")
print("Model weights saved.")