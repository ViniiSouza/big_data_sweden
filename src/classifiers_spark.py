"""Registry of Spark ML classifiers used by the experiment runner.

Each entry maps a short name (used in CLI + metrics rows) to a zero-arg factory
that returns a freshly-instantiated estimator. Adding a new classifier is one
function + one dict entry.

Hyperparameter choices for ``random_forest`` are recorded in
docs/adr/0001-rf-hyperparameters.md.
"""
from __future__ import annotations

from typing import Callable

from pyspark.ml.classification import RandomForestClassifier

from src.split import DEFAULT_SEED


def random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=DEFAULT_SEED,
    )


CLASSIFIERS: dict[str, Callable[[], object]] = {
    "random_forest": random_forest,
}
