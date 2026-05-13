"""Entry point to test reading and normalization of the 4 datasets.

Usage:
    python main.py                   # run all
    python main.py fraud_detection   # run only one by name (info.name)
"""
from __future__ import annotations

import sys

from src.datasets import (
    DatasetLoader,
    FraudDetectionLoader,
    GenZSocialLoader,
    JobSalaryLoader,
    LolMatchesLoader,
)
from src.spark_session import get_spark


LOADERS: list[type[DatasetLoader]] = [
    GenZSocialLoader,
    FraudDetectionLoader,
    LolMatchesLoader,
    JobSalaryLoader,
]


def main(selected: str | None = None) -> None:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    for loader_cls in LOADERS:
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


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
