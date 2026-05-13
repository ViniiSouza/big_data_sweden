from src.datasets.base import DatasetLoader
from src.datasets.gen_z_social import GenZSocialLoader
from src.datasets.fraud_detection import FraudDetectionLoader
from src.datasets.lol_matches import LolMatchesLoader
from src.datasets.job_salary import JobSalaryLoader

__all__ = [
    "DatasetLoader",
    "GenZSocialLoader",
    "FraudDetectionLoader",
    "LolMatchesLoader",
    "JobSalaryLoader",
]
