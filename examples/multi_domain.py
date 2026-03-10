"""Cross-domain transfer: the brain gets smarter with each task."""

import theorist

# Task 1: optimize a quadratic
@theorist.experiment(
    search_space={"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    metric="loss",
    minimize=True,
    domain="quadratic",
)
def quadratic(config):
    return {"loss": (config["x"] - 4) ** 2}

# Task 2: optimize a different function -- same structure
@theorist.experiment(
    search_space={"temp": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]},
    metric="error",
    minimize=True,
    domain="temperature",
)
def temperature(config):
    return {"error": abs(config["temp"] - 0.7) * 10}

# Run both. The brain accumulates knowledge from task 1
# and uses it to explore smarter in task 2.
r1 = quadratic.optimize(n=12)
print(f"\nQuadratic best: x={r1.best_config['x']}, loss={r1.best_metric:.4f}")

r2 = temperature.optimize(n=12)
print(f"\nTemperature best: temp={r2.best_config['temp']}, error={r2.best_metric:.4f}")

# Inspect the brain
brain = theorist.Brain()
print(f"\n{brain.summary()}")
