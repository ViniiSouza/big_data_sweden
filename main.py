"""Entry point.

Subcommands:
    smoke        Read + normalize each dataset, print schema and sample rows.
                 Used while developing the loaders' normalize().
    experiments  Train every classifier on every dataset, print metrics.

Usage:
    python main.py                                          # default: smoke, all
    python main.py smoke fraud_detection                    # smoke, single dataset
    python main.py experiments                              # all datasets x all classifiers
    python main.py experiments --dataset fraud_detection    # filter datasets
    python main.py experiments --classifier random_forest   # filter classifiers
"""
from __future__ import annotations

import argparse
import sys


def cmd_smoke(args: argparse.Namespace) -> None:
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
        if args.dataset and loader.info.name != args.dataset:
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


def cmd_experiments(args: argparse.Namespace) -> None:
    from pathlib import Path

    from src.experiments import format_table, run_all
    from src.spark_session import get_spark

    csv_path = Path("results/experiments.csv")

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        rows = run_all(
            spark,
            dataset_filter=args.dataset,
            classifier_filter=args.classifier,
            csv_path=csv_path,
        )
    finally:
        spark.stop()

    print("\n=== summary ===")
    print(format_table(rows))
    print(f"\nresults written to {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    p_smoke = sub.add_parser("smoke", help="Read + normalize datasets, print samples.")
    p_smoke.add_argument(
        "dataset", nargs="?", default=None,
        help="Optional info.name to run a single loader.",
    )
    p_smoke.set_defaults(func=cmd_smoke)

    p_exp = sub.add_parser("experiments", help="Train and evaluate classifiers.")
    p_exp.add_argument("--dataset", default=None, help="Filter by dataset info.name.")
    p_exp.add_argument("--classifier", default=None, help="Filter by classifier name.")
    p_exp.set_defaults(func=cmd_experiments)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv

    # Default to "smoke" when no subcommand is given, preserving the historical
    # `python main.py [dataset]` ergonomics.
    if not argv or argv[0] not in {"smoke", "experiments", "-h", "--help"}:
        argv = ["smoke", *argv]

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
