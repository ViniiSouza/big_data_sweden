import pandas as pd

from src.datasets_pandas.base import DatasetInfo, DatasetLoader


class LolMatchesLoader(DatasetLoader):
    """League of Legends Match Dataset 2025 — pandas baseline (stub).

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

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO (dataset owner): explode/aggregate by match if needed,
        # drop ids, fix types.
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df.dropna()
