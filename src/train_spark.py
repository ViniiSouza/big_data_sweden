"""Shared training + evaluation for Spark ML classifiers.

One function, ``train_and_evaluate``, takes already-split DataFrames and an
already-instantiated classifier (so the caller decides which one and with
which hyperparameters) and returns a flat dict of metrics. That dict is what
the experiment runner will later dump into a CSV.
"""
from __future__ import annotations

import time

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame


def train_and_evaluate(
    train_df: DataFrame,
    test_df: DataFrame,
    classifier,
) -> dict:
    """Fit ``classifier`` on ``train_df``, predict on ``test_df``, return metrics.

    Args:
        train_df / test_df: must contain ``label`` and ``features`` columns
            (output of ``build_features`` followed by ``train_test_split``).
        classifier: any ``pyspark.ml.classification`` estimator already
            instantiated (e.g. ``LogisticRegression(maxIter=20)``).

    Returns:
        dict with accuracy, f1, train_time_s, predict_time_s.
    """
    # Train
    t0 = time.perf_counter()
    model = classifier.fit(train_df)
    train_time = time.perf_counter() - t0

    # Predict — ``.count()`` forces execution because ``transform`` is lazy,
    # otherwise the time would be measured as ~0 and the actual work would
    # only happen later, inside the evaluator.
    t0 = time.perf_counter()
    preds = model.transform(test_df)
    preds.cache()
    preds.count()
    predict_time = time.perf_counter() - t0

    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(preds)

    preds.unpersist()

    return {
        "accuracy": acc,
        "f1": f1,
        "train_time_s": train_time,
        "predict_time_s": predict_time,
    }
