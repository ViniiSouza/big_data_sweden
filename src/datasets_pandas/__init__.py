from src.datasets_pandas.base import DatasetInfo, DatasetLoader
from src.datasets_pandas.gen_z_social import GenZSocialLoader
from src.datasets_pandas.fraud_detection import FraudDetectionLoader
from src.datasets_pandas.lol_matches import LolMatchesLoader
from src.datasets_pandas.job_salary import JobSalaryLoader

__all__ = [
    "DatasetInfo",
    "DatasetLoader",
    "GenZSocialLoader",
    "FraudDetectionLoader",
    "LolMatchesLoader",
    "JobSalaryLoader",
]
