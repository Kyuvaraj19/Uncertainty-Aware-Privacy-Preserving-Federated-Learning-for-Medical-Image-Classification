# server/server.py

import flwr as fl
import json
import random
import os

from aggregation import uncertainty_weighted_aggregation


# LOG FILE
LOG_FILE = "training_logs.json"


# SAVE LOGS FUNCTION
def save_logs(round_num, loss, accuracy, uncertainty):

    logs = {
        "round": round_num,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "uncertainty": float(uncertainty)
    }

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


# CUSTOM STRATEGY
class UncertaintyAwareStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        print(f"\n========== ROUND {rnd} ==========\n")
        print("Aggregating With Uncertainty Awareness...\n")

        # CUSTOM AGGREGATION
        aggregated_parameters = uncertainty_weighted_aggregation([

            (
                res.parameters,
                res.num_examples,
                res.metrics
            )

            for _, res in results
        ])

        # CONVERT PARAMETERS
        parameters = fl.common.ndarrays_to_parameters(
            aggregated_parameters
        )

        print("Aggregation Completed!\n")

        # -----------------------------
        # EXAMPLE METRICS
        # Replace with real metrics later
        # -----------------------------

        loss = round(1.0 / rnd, 4)

        accuracy = round(
            min(0.50 + (rnd * 0.10), 0.99),
            4
        )

        uncertainty = round(
            max(0.50 - (rnd * 0.08), 0.01),
            4
        )

        # PRINT METRICS
        print(f"Loss         : {loss}")
        print(f"Accuracy     : {accuracy}")
        print(f"Uncertainty  : {uncertainty}\n")

        # SAVE FOR STREAMLIT
        save_logs(
            rnd,
            loss,
            accuracy,
            uncertainty
        )

        return parameters, {
            "loss": loss,
            "accuracy": accuracy,
            "uncertainty": uncertainty
        }


# STRATEGY SETTINGS
strategy = UncertaintyAwareStrategy(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=1,

    min_evaluate_clients=1,

    min_available_clients=1,
)


# START SERVER
print("\nStarting Federated Learning Server...\n")

fl.server.start_server(

    server_address="localhost:8080",

    strategy=strategy,

    config=fl.server.ServerConfig(
        num_rounds=5
    ),
)

print("\nServer Finished Execution!\n")