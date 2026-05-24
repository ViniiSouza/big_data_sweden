from pyspark.sql import SparkSession


def get_spark(app_name: str = "BigDataSweden", master: str = "local[*]") -> SparkSession:
    """Create or return the active SparkSession.

    master='local[*]' uses all available cores (baseline/local run).
    For parallelization experiments, switch the master (e.g. 'local[2]',
    'local[4]') or point it to a real cluster.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
