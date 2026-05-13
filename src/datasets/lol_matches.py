from pyspark.sql import DataFrame

from src.datasets.base import DatasetInfo, DatasetLoader


class LolMatchesLoader(DatasetLoader):
    """League of Legends Match Dataset 2025.

    Source: https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="lol_matches",
            path="data/raw/lol_matches/league_data.csv",  # TODO: confirm CSV name after download
            target_column="win",  # TODO: confirm target column (team win)
            size_category="large",
        )

    def normalize(self, df: DataFrame) -> DataFrame:
        # TODO (dataset owner): explode/aggregate by match if needed,
        # drop ids, fix types.
        for col in df.columns:
            df = df.withColumnRenamed(col, col.strip().lower().replace(" ", "_"))
        return df.dropna()
