from pyspark.sql import SparkSession


def get_spark(
    app_name: str = "BigDataSweden",
    workers: int = 1,
    shuffle_partitions: int = 1,
) -> SparkSession:
    """Create or return the active SparkSession.

    Args:
        workers: number of cores used by Spark's local executor. Translates to
            ``local[workers]``. Default 1 (fully serial baseline).
        shuffle_partitions: value of ``spark.sql.shuffle.partitions``. Default 1
            (no parallelism after shuffles).

    The parallelization experiment varies ``workers`` and ``shuffle_partitions``
    to measure how Spark scales on a single machine.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if shuffle_partitions < 1:
        raise ValueError(f"shuffle_partitions must be >= 1, got {shuffle_partitions}")

    return (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{workers}]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
