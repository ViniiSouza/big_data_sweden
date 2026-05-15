"""Entry point to test reading and normalization of the 4 datasets.

Usage:
    python main.py                                # Spark engine (default), all datasets
    python main.py fraud_detection                # Spark, single dataset by info.name
    python main.py --engine pandas                # Non-parallel pandas baseline, all
    python main.py --engine pandas gen_z_social   # Pandas, single dataset
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


def run_pandas(selected: str | None) -> None:
    from src.datasets_pandas import (
        DatasetLoader,
        FraudDetectionLoader,
        GenZSocialLoader,
        JobSalaryLoader,
        LolMatchesLoader,
    )

    loaders: list[type[DatasetLoader]] = [
        GenZSocialLoader,
        FraudDetectionLoader,
        LolMatchesLoader,
        JobSalaryLoader,
    ]

    for loader_cls in loaders:
        loader = loader_cls()
        if selected and loader.info.name != selected:
            continue

        print(f"\n=== {loader.info.name} ({loader.info.size_category}) ===")
        print(f"path: {loader.info.path}")
        try:
            df = loader.run()
        except Exception as e:  # missing file is the usual error here
            print(f"[error] {type(e).__name__}: {e}")
            continue

        print(f"rows: {len(df)} | columns: {len(df.columns)}")
        print(f"target: {loader.info.target_column}")
        print("schema:")
        print(df.dtypes.to_string())
        print("sample:")
        print(df.head(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        choices=["spark", "pandas"],
        default="spark",
        help="Execution engine: 'spark' (parallel) or 'pandas' (non-parallel baseline).",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional info.name to run a single loader (e.g. gen_z_social).",
    )
    args = parser.parse_args()

    if args.engine == "spark":
        run_spark(args.dataset)
    else:
        run_pandas(args.dataset)


if __name__ == "__main__":
    main()
