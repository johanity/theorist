"""Theorist in 10 lines."""

import theorist

@theorist.experiment(
    search_space={"x": [-5, -3, -1, 0, 1, 3, 5]},
    metric="score",
    minimize=True,
    domain="quadratic",
)
def objective(config):
    return {"score": (config["x"] - 1) ** 2}

results = objective.optimize(n=15)
print(results.report())
