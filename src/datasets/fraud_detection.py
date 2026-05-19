from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType

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
        #clean column if no "is_fraud" or "amt" is null
        df = df.dropna(subset=["is_fraud", "amt"])

        # convert date into hour (0 to 23) and day of week (0 to 6)
        df = df.withColumn("timestamp", F.to_timestamp("trans_date_trans_time"))

        df = df.withColumn("trans_hour", F.hour("timestamp").cast(IntegerType()))
        df = df.withColumn("trans_day_of_week", F.dayofweek("timestamp").cast(IntegerType()))

        # getting age from dob 
        df = df.withColumn("dob_timestamp", F.to_timestamp("dob"))
        df = df.withColumn("age", (F.year("timestamp") - F.year("dob_timestamp")).cast(IntegerType()))

        # calculating distance between merchant and customer

        # Radius of the Earth in kilometers
        R = 6371.0

        # convert degrees to radians
        lat1 = F.radians(F.col("lat"))
        long1 = F.radians(F.col("long"))
        lat2 = F.radians(F.col("merch_lat"))
        long2 = F.radians(F.col("merch_long"))

        # differences in coordinates
        dlat = lat2 - lat1
        dlong = long2 - long1

        # Haversine formula
        a = F.sin(dlat / 2)**2 + F.cos(lat1) * F.cos(lat2) * F.sin(dlong / 2)**2
        c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
        df = df.withColumn("distance_to_merchant", (R * c).cast(DoubleType()))

        # select and rename final columns, with explicit type casts
        final_columns = [
            F.col("amt").cast(DoubleType()),
            F.col("category").cast(StringType()),
            F.col("gender").cast(StringType()),
            F.col("state").cast(StringType()),
            F.col("city_pop").cast(IntegerType()),
            F.col("job").cast(StringType()),
            F.col("trans_hour").cast(IntegerType()),
            F.col("trans_day_of_week").cast(IntegerType()),
            F.col("age").cast(IntegerType()),
            F.col("distance_to_merchant").cast(DoubleType()),
            F.col("is_fraud").cast(IntegerType())  # Ta cible (Target)
        ]

        df = df.select(final_columns)

        for col in df.columns:
            df = df.withColumnRenamed(col, col.strip().lower().replace(" ", "_"))
        return df.dropna()
