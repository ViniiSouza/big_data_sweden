from abc import ABC, abstractmethod
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession


@dataclass
class DatasetInfo:
    """Metadata for each dataset used in the project.

    name: short name used for logs/reports.
    path: CSV file (or directory) path under data/raw.
    target_column: name of the target column after normalization.
    size_category: 'small', 'medium' or 'large' (defined by the team).
    """
    name: str
    path: str
    target_column: str
    size_category: str


class DatasetLoader(ABC):
    """Base class for dataset loaders.

    Each team member subclasses this to handle one dataset.
    Standard flow:
        loader = MyLoader(spark)
        df_raw = loader.load()
        df = loader.normalize(df_raw)
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    @property
    @abstractmethod
    def info(self) -> DatasetInfo: ...

    def load(self) -> DataFrame:
        """Raw CSV read. Default covers the common case; override if needed."""
        return (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .csv(self.info.path)
        )

    @abstractmethod
    def normalize(self, df: DataFrame) -> DataFrame:
        """Dataset-specific normalization.

        Each implementation is expected to:
          - rename columns to snake_case;
          - handle missing values;
          - cast types when inferSchema is not enough;
          - ensure self.info.target_column exists in the final DataFrame.
        """
        ...

    def run(self) -> DataFrame:
        """Shortcut: load and normalize."""
        return self.normalize(self.load())
