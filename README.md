# theorist

Prediction-error optimization for Python. One file. Zero dependencies.

Most optimization loops try things and keep what works. Theorist does one thing differently: it guesses first. The prediction error is the learning signal.

```
predict → run → surprise → learn → transfer
```

## What is Theorist?

A first-principles approach to optimization. Instead of blindly searching, Theorist forces a prediction before each experiment and learns from the error. It builds a theory of *why* things work,not just *what* works,and transfers that understanding across tasks.

- **No dependencies**, pure Python, single file
- **No LLM or API keys required**, the prediction-error loop runs locally
- **Cross-domain transfer**, what it learns on one task makes the next task faster
- **Drop-in replacement** for random search or grid search with better sample efficiency

## How is this different from Optuna / Ray Tune / Bayesian optimization?

Those tools model *what* works. Theorist models *why* it works, by forcing a prediction before each experiment and learning from the error. This means:

1. It transfers knowledge across different optimization tasks
2. It adapts immediately when the problem changes (e.g., more compute, different dataset)
3. It gets better at predicting, not just searching,understanding instead of memorizing

In our experiments, this approach adapted to a 5x compute shift on the first try. Traditional search had to start over. The same pattern that makes autonomous AI agents fragile,optimizing without understanding,is what makes most hyperparameter tools plateau.

## Install

```bash
pip install theorist
```

Or just copy `theorist.py` into your project. It's one file, ~170 lines of logic.

## Quick start

```python
import theorist

@theorist.experiment(
    search_space={"lr": [1e-4, 1e-3, 1e-2], "layers": [2, 4, 8]},
    metric="val_loss",
    minimize=True,
)
def train(config):
    return {"val_loss": run_training(**config)}

results = train.optimize(n=20)
```

Brain persists at `~/.theorist/`. Second task starts smarter.

## Cross-domain transfer

```python
@theorist.experiment(search_space={...}, metric="loss", domain="training")
def task1(config): ...

@theorist.experiment(search_space={...}, metric="loss", domain="inference")
def task2(config): ...

task1.optimize(n=20)
task2.optimize(n=20)  # starts smarter,transfers what it learned
```

## Compare vs random search

```python
from theorist import compare
compare(fn=train, space={...}, n=20, metric="loss")
```

## Brain

```python
from theorist import Brain
brain = Brain()
print(brain.summary())
brain.reset()
```

## Origin

This is the SDK version of the prediction-error loop from [epistemic-autoresearch](https://github.com/johanity/epistemic-autoresearch).

251 experiments across three autonomous agent types (blind search, reflection, prediction-error). The predicting agent adapted to a 5x compute shift on its first try. The reflecting agent took 7 tries. The searching agent never caught up.

The core insight: an autonomous agent that predicts before it acts builds real understanding. One that only reacts builds nothing. This is why most AI agents are fragile,they optimize without reasoning from first principles.

Theorist packages that loop so any optimization problem can use it.

## Research

- Paper + experiments: [epistemic-autoresearch](https://github.com/johanity/epistemic-autoresearch)
- Previous study: [barcaui-predicted-karpathy](https://github.com/johanity/barcaui-predicted-karpathy)

## License

MIT,[Johan David Bonilla](https://github.com/johanity)
