from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from src.datasets.base import DatasetInfo, DatasetLoader


class GenZSocialLoader(DatasetLoader):
    """Gen Z Social Media Usage Dataset.

    Source: https://www.kaggle.com/datasets/sharmajicoder/gen-z-social-media-usage-dataset

    Raw schema (12 columns, no nulls observed):
        age, gender, country, daily_usage_hours, primary_platform,
        num_platforms_used, purpose, avg_session_minutes, night_usage,
        mental_health_score, addiction_level, screen_time_before_sleep

    Target: ``addiction_level`` (Low / Medium / High) — 3-class classification.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="gen_z_social",
            path="data/raw/gen_z_social/genz_social_media_usage_1M.csv",
            target_column="addiction_level",
            size_category="medium",
        )

    def normalize(self, df: DataFrame) -> DataFrame:
        """Normalization decisions for this dataset:

        1. Column names are already snake_case in the CSV — nothing to rename.
        2. No columns are dropped: all 12 are candidate features or the target.
        3. Explicit type casts so the downstream feature pipeline can rely on
           them (do not trust inferSchema across team members / configs).
        4. Drop rows with null target. The raw file has no nulls today, but
           this guards future re-downloads.
        5. Keep only the expected columns at the end (locks the output contract).
        """
        numeric_double = [
            "daily_usage_hours",
            "avg_session_minutes",
            "mental_health_score",
            "screen_time_before_sleep",
        ]
        numeric_int = ["age", "num_platforms_used", "night_usage"]
        categorical = ["gender", "country", "primary_platform", "purpose"]
        target = self.info.target_column

        for c in numeric_double:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
        for c in numeric_int:
            df = df.withColumn(c, F.col(c).cast(IntegerType()))
        for c in categorical + [target]:
            df = df.withColumn(c, F.trim(F.col(c).cast("string")))

        df = df.dropna(subset=[target])
        df = df.filter(F.col(target).isin("Low", "Medium", "High"))

        return df.select(*numeric_int, *numeric_double, *categorical, target)
