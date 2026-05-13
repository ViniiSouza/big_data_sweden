from pyspark.sql import DataFrame

from src.datasets.base import DatasetInfo, DatasetLoader


class FraudDetectionLoader(DatasetLoader):
    """Credit Card Fraud Detection.

    Source: https://www.kaggle.com/datasets/kartik2112/fraud-detection
    We use fraudTrain.csv as the main file (large dataset).
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="fraud_detection",
            path="data/raw/fraud_detection/fraudTrain.csv",
            target_column="is_fraud",
            size_category="large",
        )

    def normalize(self, df: DataFrame) -> DataFrame:
        # TODO (dataset owner): drop irrelevant columns (e.g. _c0, trans_num,
        # first/last name), handle dates and categorical fields.
        for col in df.columns:
            df = df.withColumnRenamed(col, col.strip().lower().replace(" ", "_"))
        return df.dropna()
