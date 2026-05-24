"""Entry point to test reading and normalization of the 4 datasets.

Usage:
    python main.py                  # All datasets
    python main.py fraud_detection  # Single dataset by info.name
"""
from __future__ import annotations

import argparse


def run_spark(selected: str | None) -> None:
    from src.datasets import (
        DatasetLoader,
        FraudDetectionLoader,
        GenZSocialLoader,
        JobSalaryLoader,
        LolMatchesLoader,
    )
    from src.spark_session import get_spark

    loaders: list[type[DatasetLoader]] = [
        GenZSocialLoader,
        FraudDetectionLoader,
        LolMatchesLoader,
        JobSalaryLoader,
    ]

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    for loader_cls in loaders:
        loader = loader_cls(spark)
        if selected and loader.info.name != selected:
            continue

        print(f"\n=== {loader.info.name} ({loader.info.size_category}) ===")
        print(f"path: {loader.info.path}")
        try:
            df = loader.run()
        except Exception as e:  # missing file is the usual error here
            print(f"[error] {type(e).__name__}: {e}")
            continue

        print(f"rows: {df.count()} | columns: {len(df.columns)}")
        print(f"target: {loader.info.target_column}")
        df.printSchema()
        df.show(5, truncate=False)

    spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional info.name to run a single loader (e.g. gen_z_social).",
    )
    args = parser.parse_args()

    run_spark(args.dataset)


if __name__ == "__main__":
    main()
