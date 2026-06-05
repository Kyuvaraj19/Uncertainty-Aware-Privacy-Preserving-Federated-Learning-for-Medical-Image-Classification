# client/client.py

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import os

from model import CNN
from dataset import load_data
from uncertainty import compute_uncertainty


# ==========================================
# DEVICE SETUP
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {DEVICE}\n")


# ==========================================
# LOAD MODEL
# ==========================================

model = CNN(num_classes=2).to(DEVICE)

print("CNN Model Loaded Successfully!\n")


# ==========================================
# LOAD DATA
# ==========================================

trainloader, testloader = load_data()


# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# ROUND COUNTER
# ==========================================

round_counter = 0


# ==========================================
# TRAIN FUNCTION
# ==========================================

def train(model, loader, epochs=1):

    print("\nStarting Local Training...\n")

    model.train()

    final_loss = 0

    for epoch in range(epochs):

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # PRINT EVERY 5 BATCHES
            if batch_idx % 5 == 0:

                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}] "
                    f"Loss: {loss.item():.4f}"
                )

        final_loss = running_loss / len(loader)

        print(
            f"\nEpoch {epoch+1} Completed "
            f"| Average Loss: {final_loss:.4f}\n"
        )

    print("Local Training Finished!\n")

    return final_loss


# ==========================================
# GET MODEL PARAMETERS
# ==========================================

def get_parameters():

    return [
        val.cpu().numpy()
        for val in model.state_dict().values()
    ]


# ==========================================
# SET MODEL PARAMETERS
# ==========================================

def set_parameters(parameters):

    params_dict = zip(
        model.state_dict().keys(),
        parameters
    )

    state_dict = {
        k: torch.tensor(v)
        for k, v in params_dict
    }

    model.load_state_dict(
        state_dict,
        strict=True
    )


# ==========================================
# FLOWER CLIENT
# ==========================================

class HospitalClient(fl.client.NumPyClient):


    # ======================================
    # SEND INITIAL PARAMETERS
    # ======================================

    def get_parameters(self, config):

        print("\nSending Initial Parameters To Server...\n")

        return get_parameters()


    # ======================================
    # TRAINING ROUND
    # ======================================

    def fit(self, parameters, config):

        global round_counter

        print("\n==============================")
        print("NEW FEDERATED ROUND STARTED")
        print("==============================\n")

        # RECEIVE GLOBAL MODEL
        set_parameters(parameters)

        print("Global Model Received From Server\n")

        # LOCAL TRAINING
        loss = train(
            model,
            trainloader,
            epochs=1
        )

        # ==================================
        # COMPUTE UNCERTAINTY
        # ==================================

        uncertainty = compute_uncertainty(
            model,
            testloader,
            DEVICE
        )

        # ==================================
        # EVALUATE ACCURACY
        # ==================================

        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():

            for images, labels in testloader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        accuracy = correct / total

        print(f"Accuracy : {accuracy:.4f}")

        # ==================================
        # UPDATE ROUND
        # ==================================

        round_counter += 1

        # ==================================
        # SAVE TRAINING LOGS
        # ==================================

        log_data = {
            "round": [round_counter],
            "loss": [loss],
            "accuracy": [accuracy],
            "uncertainty": [uncertainty]
        }

        df = pd.DataFrame(log_data)

        if os.path.exists("training_logs.csv"):

            df.to_csv(
                "training_logs.csv",
                mode="a",
                header=False,
                index=False
            )

        else:

            df.to_csv(
                "training_logs.csv",
                index=False
            )

        print("\nTraining Logs Saved Successfully!\n")

        print(
            f"Sending Updated Weights "
            f"With Uncertainty: {uncertainty:.4f}\n"
        )

        return (
            get_parameters(),
            len(trainloader.dataset),
            {"uncertainty": uncertainty}
        )


    # ======================================
    # EVALUATION
    # ======================================

    def evaluate(self, parameters, config):

        print("\nEvaluating Global Model...\n")

        set_parameters(parameters)

        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():

            for images, labels in testloader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        accuracy = correct / total

        print(f"Evaluation Accuracy : {accuracy:.4f}\n")

        return (
            float(accuracy),
            len(testloader.dataset),
            {}
        )


# ==========================================
# START CLIENT
# ==========================================

print("Connecting To Federated Server...\n")

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=HospitalClient()
)

print("\nClient Finished Execution!\n")