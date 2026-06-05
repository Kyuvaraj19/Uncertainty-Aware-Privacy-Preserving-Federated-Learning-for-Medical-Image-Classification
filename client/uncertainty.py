# client/uncertainty.py

import torch
import torch.nn.functional as F


def compute_uncertainty(model, dataloader, device):

    print("\nCalculating Uncertainty...\n")

    model.eval()

    total_entropy = 0.0
    count = 0

    with torch.no_grad():

        for images, _ in dataloader:

            images = images.to(device)

            outputs = model(images)

            # Convert outputs to probabilities
            probs = F.softmax(outputs, dim=1)

            # Entropy calculation
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-8),
                dim=1
            )

            total_entropy += entropy.sum().item()
            count += entropy.size(0)

    avg_uncertainty = total_entropy / count

    print(f"Average Uncertainty : {avg_uncertainty:.4f}")

    print("\nUncertainty Calculation Completed!\n")

    return avg_uncertainty