import numpy as np
import pandas as pd

from src.datasets_pandas.base import DatasetInfo, DatasetLoader


class JobSalaryLoader(DatasetLoader):
    """Job Salary Prediction Dataset — pandas baseline.

    Mirrors src.datasets.job_salary.JobSalaryLoader. Terciles are computed
    exactly with pandas.Series.quantile, the natural equivalent of Spark's
    approxQuantile(..., relativeError=0.01). On ~250k rows the boundary
    difference is negligible.
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="job_salary",
            path="data/raw/job_salary/job_salary_prediction_dataset.csv",
            target_column="salary_category",
            size_category="medium",
        )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
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

        df["salary"] = pd.to_numeric(df["salary"], errors="coerce").astype("float64")
        df = df.dropna(subset=["salary"])

        q33, q66 = df["salary"].quantile([1 / 3, 2 / 3]).tolist()

        df[target] = np.where(
            df["salary"] <= q33, "Low",
            np.where(df["salary"] <= q66, "Medium", "High"),
        )

        for c in numeric_int:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in categorical + [target]:
            df[c] = df[c].astype("string").str.strip()

        df = df.dropna(subset=[target])
        df = df[df[target].isin(["Low", "Medium", "High"])]

        return df[[*numeric_int, *categorical, target]].reset_index(drop=True)
