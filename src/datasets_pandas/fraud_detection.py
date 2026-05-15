import pandas as pd

from src.datasets_pandas.base import DatasetInfo, DatasetLoader


class FraudDetectionLoader(DatasetLoader):
    """Credit Card Fraud Detection — pandas baseline (stub).

    Source: https://www.kaggle.com/datasets/kartik2112/fraud-detection
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="fraud_detection",
            path="data/raw/fraud_detection/fraudTrain.csv",
            target_column="is_fraud",
            size_category="large",
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO (dataset owner): drop irrelevant columns (e.g. _c0, trans_num,
        # first/last name), handle dates and categorical fields.
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df.dropna()
