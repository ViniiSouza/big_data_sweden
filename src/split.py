"""Shared train/test split.

Notes:
    Spark's ``randomSplit`` is a plain random split, not stratified. For our
    datasets (3-class targets with reasonably balanced classes) this is enough.
    If a future dataset has heavy class imbalance, replace this helper with a
    stratified variant rather than working around it in the loaders.
"""
from __future__ import annotations

from pyspark.sql import DataFrame

DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8


def train_test_split(
    df: DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[DataFrame, DataFrame]:
    """Split a Spark DataFrame into (train, test) using a fixed seed."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    return train_df, test_df
