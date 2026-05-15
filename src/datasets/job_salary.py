from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

from src.datasets.base import DatasetInfo, DatasetLoader


class JobSalaryLoader(DatasetLoader):
    """Job Salary Prediction Dataset.

    Source: https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset

    Raw schema (10 columns):
        job_title, experience_years, education_level, skills_count, industry,
        company_size, location, remote_work, certifications, salary

    Target: ``salary_category`` (Low / Medium / High) — 3-class classification
    derived from the continuous ``salary`` column by quantile bucketing
    (terciles), so the resulting target is roughly balanced.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="job_salary",
            path="data/raw/job_salary/job_salary_prediction_dataset.csv",
            target_column="salary_category",
            size_category="medium",
        )

    def normalize(self, df: DataFrame) -> DataFrame:
        """Normalization decisions for this dataset:

        1. Column names are already snake_case in the CSV — nothing to rename.
        2. ``salary`` is the raw numeric label; we discretize it into 3 classes
           (Low/Medium/High) using terciles computed with ``approxQuantile``
           and drop the original column to avoid target leakage downstream.
        3. Explicit type casts so the downstream feature pipeline can rely on
           them (do not trust inferSchema across team members / configs).
        4. Drop rows with null target (and null salary before bucketing).
        5. Keep only the expected columns at the end (locks the output contract).
        """
        numeric_int = ["experience_years", "skills_count", "certifications"]
        categorical = [
            "job_title",
            "education_level",
            "industry",
            "company_size",
            "location",
            "remote_work",
        ]
        target = self.info.target_column

        df = df.withColumn("salary", F.col("salary").cast(DoubleType()))
        df = df.dropna(subset=["salary"])

        # Terciles — 0.01 relativeError is a good balance of accuracy and cost
        # for a single ~250k-row dataset (exact quantiles would force a full
        # sort that doesn't pay off here).
        q33, q66 = df.approxQuantile("salary", [1 / 3, 2 / 3], 0.01)

        df = df.withColumn(
            target,
            F.when(F.col("salary") <= q33, "Low")
            .when(F.col("salary") <= q66, "Medium")
            .otherwise("High"),
        )

        for c in numeric_int:
            df = df.withColumn(c, F.col(c).cast(IntegerType()))
        for c in categorical + [target]:
            df = df.withColumn(c, F.trim(F.col(c).cast("string")))

        df = df.dropna(subset=[target])
        df = df.filter(F.col(target).isin("Low", "Medium", "High"))

        return df.select(*numeric_int, *categorical, target)
