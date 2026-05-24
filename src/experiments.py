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

import csv
from pathlib import Path

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

CSV_FIELDS = [
    "dataset", "classifier", "workers", "partitions",
    "accuracy", "f1", "train_time_s", "predict_time_s", "total_time_s",
]


def run_all(
    spark: SparkSession,
    dataset_filter: str | None = None,
    classifier_filter: str | None = None,
    workers: int = 1,
    partitions: int = 1,
    csv_path: Path | None = None,
) -> list[dict]:
    """Run every (dataset, classifier) combination and return metrics rows.

    Args:
        spark: active SparkSession.
        dataset_filter: if set, only the loader whose ``info.name`` matches runs.
        classifier_filter: if set, only the matching key from ``CLASSIFIERS`` runs.
        workers: number of Spark workers in use; recorded in each metrics row
            for downstream comparison. The caller is responsible for matching
            this to what was passed to ``get_spark``.
        partitions: target partition count for the dataset DataFrame after
            load. Each dataset is ``repartition(partitions)``-ed before
            features are built, so the parallelism of the whole pipeline
            (features, split, fit) is bounded by it. Default 1 forces serial.
        csv_path: if set, each metrics row is appended to this CSV as it
            completes (header written first). Survives a mid-run crash.

    Returns:
        List of metrics rows. Each row also gets printed to stdout as it
        completes (streaming progress).
    """
    if partitions < 1:
        raise ValueError(f"partitions must be >= 1, got {partitions}")
    if classifier_filter and classifier_filter not in CLASSIFIERS:
        raise ValueError(
            f"unknown classifier {classifier_filter!r}; "
            f"available: {sorted(CLASSIFIERS)}"
        )

    classifier_names = (
        [classifier_filter] if classifier_filter else list(CLASSIFIERS)
    )

    csv_file = None
    csv_writer = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        csv_file.flush()

    rows: list[dict] = []
    try:
        for loader_cls in LOADERS:
            loader = loader_cls(spark)
            ds_name = loader.info.name
            if dataset_filter and ds_name != dataset_filter:
                continue

            print(f"\n=== dataset: {ds_name} ({loader.info.size_category}) ===")
            try:
                df = loader.run().repartition(partitions)
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

                    total_time = metrics["train_time_s"] + metrics["predict_time_s"]
                    row = {
                        "dataset": ds_name,
                        "classifier": clf_name,
                        "workers": workers,
                        "partitions": partitions,
                        **metrics,
                        "total_time_s": total_time,
                    }
                    rows.append(row)
                    if csv_writer is not None:
                        csv_writer.writerow(row)
                        csv_file.flush()
                    print(
                        f"     acc={metrics['accuracy']:.4f} "
                        f"f1={metrics['f1']:.4f} "
                        f"train={metrics['train_time_s']:.2f}s "
                        f"predict={metrics['predict_time_s']:.2f}s "
                        f"total={total_time:.2f}s"
                    )
            finally:
                feat_df.unpersist()
    finally:
        if csv_file is not None:
            csv_file.close()

    return rows


def format_table(rows: list[dict]) -> str:
    """Render the accumulated metrics rows as a fixed-width table."""
    if not rows:
        return "(no results)"

    headers = [
        "dataset", "classifier", "workers", "partitions",
        "accuracy", "f1", "train_s", "predict_s", "total_s",
    ]
    formatted: list[list[str]] = [headers]
    for r in rows:
        formatted.append([
            r["dataset"],
            r["classifier"],
            str(r["workers"]),
            str(r["partitions"]),
            f"{r['accuracy']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['train_time_s']:.2f}",
            f"{r['predict_time_s']:.2f}",
            f"{r['total_time_s']:.2f}",
        ])

    widths = [max(len(row[i]) for row in formatted) for i in range(len(headers))]
    sep = "  "

    def fmt(row: list[str]) -> str:
        return sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt(formatted[0]), sep.join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in formatted[1:])
    return "\n".join(lines)
