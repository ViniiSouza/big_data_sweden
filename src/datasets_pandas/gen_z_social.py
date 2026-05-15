import pandas as pd

from src.datasets_pandas.base import DatasetInfo, DatasetLoader


class GenZSocialLoader(DatasetLoader):
    """Gen Z Social Media Usage Dataset — pandas baseline.

    Mirrors src.datasets.gen_z_social.GenZSocialLoader without parallelism.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="gen_z_social",
            path="data/raw/gen_z_social/genz_social_media_usage_1M.csv",
            target_column="addiction_level",
            size_category="medium",
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
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
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        for c in numeric_int:
            # Int64 is nullable, matching Spark IntegerType semantics.
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in categorical + [target]:
            df[c] = df[c].astype("string").str.strip()

        df = df.dropna(subset=[target])
        df = df[df[target].isin(["Low", "Medium", "High"])]

        return df[[*numeric_int, *numeric_double, *categorical, target]].reset_index(drop=True)
