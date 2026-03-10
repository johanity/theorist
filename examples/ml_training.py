"""ML hyperparameter optimization with Theorist.

This example simulates training -- replace the body with your real
training loop and Theorist handles the search.
"""

import math
import random
import theorist


@theorist.experiment(
    search_space={
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "batch_size": [16, 32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
        "layers": [2, 4, 6, 8],
    },
    metric="val_loss",
    minimize=True,
    domain="ml_training",
)
def train(config):
    """Simulated training -- replace with your real training code."""
    lr = config["lr"]
    bs = config["batch_size"]
    dropout = config["dropout"]
    layers = config["layers"]

    # Simulated loss landscape with noise
    lr_penalty = (math.log10(lr) + 3) ** 2  # optimal around 1e-3
    bs_effect = abs(math.log2(bs) - 5.5) * 0.3  # optimal around 48
    dropout_effect = abs(dropout - 0.2) * 2  # optimal around 0.2
    layer_effect = abs(layers - 4) * 0.15  # optimal around 4
    noise = random.gauss(0, 0.05)

    val_loss = 0.5 + lr_penalty * 0.1 + bs_effect + dropout_effect + layer_effect + noise
    return {"val_loss": max(val_loss, 0.01)}


if __name__ == "__main__":
    results = train.optimize(n=25)
    print(results.report())
