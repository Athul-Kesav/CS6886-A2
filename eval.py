import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def get_base_model(weights_path, device, num_classes=10):

    model = mobilenet_v2(weights=None, width_mult=1.0)

    model.features[0][0].stride = (1, 1) # type: ignore
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total