# server/test_global_model.py
import torch
from client.model import CNN
from client.dataset import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=2).to(DEVICE)
_, testloader = load_data()

def test(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

accuracy = test(model, testloader)
print(f" Global Model Accuracy: {accuracy:.4f}")
