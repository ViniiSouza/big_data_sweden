"""Experiment runner.

One experiment = one (dataset, classifier) pair. Running it means:
load -> normalize -> build features -> train/test split -> fit -> evaluate.
It produces one metrics row (see CONTEXT.md).

The loop is dataset-outer / classifier-inner so the heavy work — load,
normalize, feature pipeline, split — runs once per dataset and is reused
across classifiers. ``feat_df`` is persisted with ``MEMORY_AND_DISK`` so large
feature vectors (e.g. lol_matches after one-hot of ~150 champion columns)
spill cleanly instead of triggering MemoryStore warnings.

Skip-and-continue policy: a missing file or a fit error logs the failure and
moves on. Partial results are still useful for the report.
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

from src.classifiers_spark import CLASSIFIERS
from src.datasets import (
    DatasetLoader,
    FraudDetectionLoader,
    GenZSocialLoader,
    JobSalaryLoader,
    LolMatchesLoader,
)
from src.features import build_features
from src.split import train_test_split
from src.train_spark import train_and_evaluate

LOADERS: list[type[DatasetLoader]] = [
    GenZSocialLoader,
    FraudDetectionLoader,
    LolMatchesLoader,
    JobSalaryLoader,
]


def run_all(
    spark: SparkSession,
    dataset_filter: str | None = None,
    classifier_filter: str | None = None,
) -> list[dict]:
    """Run every (dataset, classifier) combination and return metrics rows.

    Args:
        spark: active SparkSession.
        dataset_filter: if set, only the loader whose ``info.name`` matches runs.
        classifier_filter: if set, only the matching key from ``CLASSIFIERS`` runs.

    Returns:
        List of metrics rows. Each row also gets printed to stdout as it
        completes (streaming progress).
    """
    if classifier_filter and classifier_filter not in CLASSIFIERS:
        raise ValueError(
            f"unknown classifier {classifier_filter!r}; "
            f"available: {sorted(CLASSIFIERS)}"
        )

    classifier_names = (
        [classifier_filter] if classifier_filter else list(CLASSIFIERS)
    )

    rows: list[dict] = []
    for loader_cls in LOADERS:
        loader = loader_cls(spark)
        ds_name = loader.info.name
        if dataset_filter and ds_name != dataset_filter:
            continue

        print(f"\n=== dataset: {ds_name} ({loader.info.size_category}) ===")
        try:
            df = loader.run()
            feat_df, _ = build_features(df, target_col=loader.info.target_column)
            feat_df.persist(StorageLevel.MEMORY_AND_DISK)
            train_df, test_df = train_test_split(feat_df)
        except Exception as e:
            print(f"[error] dataset {ds_name}: {type(e).__name__}: {e}")
            continue

        try:
            for clf_name in classifier_names:
                clf = CLASSIFIERS[clf_name]()
                print(f"  -> {clf_name} ...", flush=True)
                try:
                    metrics = train_and_evaluate(train_df, test_df, clf)
                except Exception as e:
                    print(f"  [error] {clf_name}: {type(e).__name__}: {e}")
                    continue

                row = {"dataset": ds_name, "classifier": clf_name, **metrics}
                rows.append(row)
                print(
                    f"     acc={metrics['accuracy']:.4f} "
                    f"f1={metrics['f1']:.4f} "
                    f"train={metrics['train_time_s']:.2f}s "
                    f"predict={metrics['predict_time_s']:.2f}s"
                )
        finally:
            feat_df.unpersist()

    return rows


def format_table(rows: list[dict]) -> str:
    """Render the accumulated metrics rows as a fixed-width table."""
    if not rows:
        return "(no results)"

    headers = ["dataset", "classifier", "accuracy", "f1", "train_s", "predict_s"]
    formatted: list[list[str]] = [headers]
    for r in rows:
        formatted.append([
            r["dataset"],
            r["classifier"],
            f"{r['accuracy']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['train_time_s']:.2f}",
            f"{r['predict_time_s']:.2f}",
        ])

    widths = [max(len(row[i]) for row in formatted) for i in range(len(headers))]
    sep = "  "

    def fmt(row: list[str]) -> str:
        return sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt(formatted[0]), sep.join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in formatted[1:])
    return "\n".join(lines)
