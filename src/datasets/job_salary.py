from pyspark.sql import DataFrame

from src.datasets.base import DatasetInfo, DatasetLoader


class JobSalaryLoader(DatasetLoader):
    """Job Salary Prediction Dataset.

    Source: https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="job_salary",
            path="data/raw/job_salary/job_salary_prediction_dataset.csv",  # TODO: confirm CSV name
            target_column="salary_category",  # TODO: define target (discretized salary bucket)
            size_category="medium",
        )

    def normalize(self, df: DataFrame) -> DataFrame:
        # TODO (dataset owner): handle free text (job_description),
        # bucket salary into categories, drop nulls.
        for col in df.columns:
            df = df.withColumnRenamed(col, col.strip().lower().replace(" ", "_"))
        return df.dropna()
