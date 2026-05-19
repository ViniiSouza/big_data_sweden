import pandas as pd
import numpy as np

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
        # to be sure to not modified original df, we create a copy here
        df = df.copy()
        
        #clean column if no "is_fraud" or "amt" is null
        df = df.dropna(subset=["is_fraud", "amt"])

        # convert date into hour (0 to 23) and day of week (0 to 6)
        timestamp = pd.to_datetime(df["trans_date_trans_time"])

        df["trans_hour"] = timestamp.dt.hour
        df["trans_day_of_week"] = timestamp.dt.dayofweek + 1 # pandas: Monday=0, Spark: Monday=1

        # getting age from dob
        dob_timestamp = pd.to_datetime(df["dob"])
        df["age"] = timestamp.dt.year - dob_timestamp.dt.year

        # calculating distance between merchant and customer using Haversine formula
        # Radius of the Earth in kilometers
        R = 6371.0

         # convert degrees to radians
        lat1 = np.radians(df["lat"])
        long1 = np.radians(df["long"])
        lat2 = np.radians(df["merch_lat"])
        long2 = np.radians(df["merch_long"])

        # differences in coordinates
        dlat = lat2 - lat1
        dlong = long2 - long1

        # Haversine formula
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        df["distance_to_merchant"] = R * c

        # select and rename final columns, with explicit type casts
        columns_mapping = {
            "amt": "float64",
            "category": "string",
            "gender": "string",
            "state": "string",
            "city_pop": "Int64",
            "job": "string",
            "trans_hour": "Int64",
            "trans_day_of_week": "Int64",
            "age": "Int64",
            "distance_to_merchant": "float64",
            "is_fraud": "Int64"  # Ta cible (Target)
        }

        df = df[list(columns_mapping.keys())]
        
        # Application stricte des types
        df = df.astype(columns_mapping)

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df.dropna()
