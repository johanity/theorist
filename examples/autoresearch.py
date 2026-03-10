"""Drop-in replacement for autoresearch's random mutation loop.

Before (autoresearch):
    while True:
        mutate random hyperparams
        train
        if improved: keep
        else: discard

After (theorist):
    @theorist.experiment(search_space={...})
    def train(config):
        return {"val_loss": run_training(config)}
    results = train.optimize(n=20)

Same loop. But theorist predicts before each run,
measures surprise after, and builds a theory that transfers.
"""

import random

import theorist


@theorist.experiment(
    search_space={
        "n_layer": [2, 3, 4, 5, 6, 8],
        "n_head": [2, 3, 4, 6, 8],
        "n_embd": [96, 128, 192, 256, 384],
        "dropout": [0.0, 0.05, 0.1, 0.15, 0.2],
        "learning_rate": [1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 3e-3],
        "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
        "batch_size": [2, 4, 8, 16, 32],
    },
    metric="val_loss",
    minimize=True,
    domain="gpt_training",
)
def train_gpt(config):
    """Simulate a GPT training run.
    Replace this with your actual training code.
    """
    # Simulated loss based on known patterns from Karp experiment:
    # - Small models (128-192 embd) beat large at short budgets
    # - Moderate LR (~1e-3 to 2e-3) beats extremes
    # - No regularization is best for short runs
    # - Depth 4-5 is optimal

    base = 4.0

    # Architecture effects
    embd = config["n_embd"]
    if embd <= 192:
        base -= 0.5  # small models win
    elif embd >= 384:
        base += 0.3  # too large, undertrained

    layers = config["n_layer"]
    if 4 <= layers <= 5:
        base -= 0.3
    elif layers >= 8:
        base += 0.2

    # LR effects
    lr = config["learning_rate"]
    if 1e-3 <= lr <= 2e-3:
        base -= 0.4
    elif lr <= 3e-4:
        base += 0.1
    elif lr >= 3e-3:
        base += 0.2

    # Regularization (harmful at short budget)
    if config["dropout"] > 0:
        base += config["dropout"] * 2
    if config["weight_decay"] > 0:
        base += config["weight_decay"]

    # Batch size (4-8 optimal)
    bs = config["batch_size"]
    if bs in (4, 8):
        base -= 0.2
    elif bs >= 16:
        base += 0.5

    # Noise
    base += random.gauss(0, 0.05)

    return {"val_loss": max(base, 2.5)}


if __name__ == "__main__":
    # Compare theorist vs blind search
    print("Running side-by-side comparison...\n")
    theorist.compare(
        fn=train_gpt,
        search_space=train_gpt._search_space,
        n=30,
        metric="val_loss",
        minimize=True,
        domain="gpt_training",
    )
