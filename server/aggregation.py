# server/aggregation.py

import flwr as fl
import numpy as np


def uncertainty_weighted_aggregation(results):
    """
    results:
    list of (parameters, num_samples, metrics)
    """

    weighted_params = None

    total_weight = 0.0

    for params, num_samples, metrics in results:

        # CONVERT FLOWER PARAMETERS TO NUMPY ARRAYS
        ndarrays = fl.common.parameters_to_ndarrays(params)

        # GET UNCERTAINTY
        uncertainty = metrics.get("uncertainty", 1.0)

        # AVOID DIVISION BY ZERO
        weight = 1.0 / (uncertainty + 1e-8)

        print(
            f"Client Uncertainty: {uncertainty:.4f}"
        )

        print(
            f"Aggregation Weight: {weight:.4f}\n"
        )

        # FIRST CLIENT
        if weighted_params is None:

            weighted_params = [
                weight * layer
                for layer in ndarrays
            ]

        # OTHER CLIENTS
        else:

            for i in range(len(ndarrays)):

                weighted_params[i] += (
                    weight * ndarrays[i]
                )

        total_weight += weight

    # NORMALIZE PARAMETERS
    aggregated_params = [
        layer / total_weight
        for layer in weighted_params
    ]

    print("Aggregation Completed Successfully!\n")

    return aggregated_params