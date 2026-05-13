from pyspark.sql import DataFrame

from src.datasets.base import DatasetInfo, DatasetLoader


class GenZSocialLoader(DatasetLoader):
    """Gen Z Social Media Usage Dataset.

    Source: https://www.kaggle.com/datasets/sharmajicoder/gen-z-social-media-usage-dataset
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="gen_z_social",
            path="data/raw/gen_z_social/gen_z_social_media_usage.csv",
            target_column="addiction_level",  # TODO: confirm final name
            size_category="small",
        )

    def normalize(self, df: DataFrame) -> DataFrame:
        # TODO (dataset owner): rename columns to snake_case, handle nulls,
        # cast types and define the final target column.
        for col in df.columns:
            df = df.withColumnRenamed(col, col.strip().lower().replace(" ", "_"))
        return df.dropna()
