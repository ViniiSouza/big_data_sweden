"""Shared feature-engineering pipeline.

Turns a normalized DataFrame (output of any ``DatasetLoader.run()``) into the
canonical ML input shape used across every experiment:

    - a numeric ``label`` column (from the string target)
    - a vector ``features`` column (numeric cols + one-hot encoded categoricals)

The same pipeline is reused by every classifier and every Spark configuration
so the comparison between experiments stays fair.
"""
from __future__ import annotations

from typing import Sequence

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType, StringType


def _infer_columns(
    df: DataFrame, target_col: str
) -> tuple[list[str], list[str]]:
    """Split columns into (numeric, categorical) based on Spark types,
    skipping the target column."""
    numeric: list[str] = []
    categorical: list[str] = []
    for field in df.schema.fields:
        if field.name == target_col:
            continue
        if isinstance(field.dataType, NumericType):
            numeric.append(field.name)
        elif isinstance(field.dataType, StringType):
            categorical.append(field.name)
        # other types (date, vector, ...) are ignored on purpose — the dataset
        # owner should have cast them in normalize().
    return numeric, categorical


def build_features(
    df: DataFrame,
    target_col: str,
    numeric_cols: Sequence[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
) -> tuple[DataFrame, PipelineModel]:
    """Fit and apply the feature pipeline.

    Args:
        df: normalized DataFrame from a ``DatasetLoader``.
        target_col: name of the string target column.
        numeric_cols / categorical_cols: optional explicit lists. If omitted,
            they are inferred from the Spark schema.

    Returns:
        (transformed_df, fitted_pipeline) where ``transformed_df`` contains at
        least ``label`` and ``features``.
    """
    inferred_num, inferred_cat = _infer_columns(df, target_col)
    numeric_cols = list(numeric_cols) if numeric_cols is not None else inferred_num
    categorical_cols = (
        list(categorical_cols) if categorical_cols is not None else inferred_cat
    )

    stages: list = []

    stages.append(
        StringIndexer(
            inputCol=target_col, outputCol="label", handleInvalid="keep"
        )
    )

    cat_idx_cols = [f"{c}_idx" for c in categorical_cols]
    cat_ohe_cols = [f"{c}_ohe" for c in categorical_cols]
    if categorical_cols:
        stages.append(
            StringIndexer(
                inputCols=list(categorical_cols),
                outputCols=cat_idx_cols,
                handleInvalid="keep",
            )
        )
        stages.append(
            OneHotEncoder(inputCols=cat_idx_cols, outputCols=cat_ohe_cols)
        )

    stages.append(
        VectorAssembler(
            inputCols=numeric_cols + cat_ohe_cols,
            outputCol="features",
            handleInvalid="skip",
        )
    )

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    return model.transform(df), model
