import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_learning_curves(csv_path="training_metrics.csv"):
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(df['epoch'], df['test_loss'], label='Test Loss', marker='o')
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()
    
    # Accuracy curves
    ax2.plot(df['epoch'], df['train_acc'] * 100, label='Train Acc (%)', marker='o')
    ax2.plot(df['epoch'], df['test_acc'] * 100, label='Test Acc (%)', marker='o')
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("baseline_learning_curves.png", dpi=300)
    plt.close()
    print("Saved: baseline_learning_curves.png")

def generate_confusion_matrix(npz_path="test_predictions.npz"):
    data = np.load(npz_path)
    preds = data['preds']
    labels = data['labels']
    
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("CIFAR-10 Confusion Matrix (Failure Modes)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved: confusion_matrix.png")

if __name__ == "__main__":
    generate_learning_curves()
    generate_confusion_matrix()