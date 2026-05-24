# Big Data Sweden — Halmstad Project

Big Data course project: supervised classification with Spark on datasets of different sizes.

## Structure

```
src/
  spark_session.py        # Shared SparkSession factory
  datasets/               # Spark loaders
    base.py               # DatasetLoader (abstract) + DatasetInfo
    gen_z_social.py       # small/medium
    fraud_detection.py    # large
    lol_matches.py        # large
    job_salary.py         # small/medium
main.py                   # Reads and normalizes every dataset
data/raw/                 # CSVs downloaded from Kaggle (not versioned)
```

## Setup

The project uses a conda environment (per the professor's recommendation) that ships with **OpenJDK 17** and **PySpark 3.5**:

```bash
conda create -n bigdata python=3.13 openjdk=17 -c conda-forge
conda activate bigdata
pip install -r requirements.txt
```

Always run `conda activate bigdata` before executing anything in this repo — running with a system JDK other than 17 will fail (Spark 3.5/4 does not support newer JDKs).

## Datasets

Download from Kaggle and place under `data/raw/<name>/`:

| Loader | Target folder | Source |
|---|---|---|
| `GenZSocialLoader`    | `data/raw/gen_z_social/`    | https://www.kaggle.com/datasets/sharmajicoder/gen-z-social-media-usage-dataset |
| `FraudDetectionLoader`| `data/raw/fraud_detection/` | https://www.kaggle.com/datasets/kartik2112/fraud-detection |
| `LolMatchesLoader`    | `data/raw/lol_matches/`     | https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025 |
| `JobSalaryLoader`     | `data/raw/job_salary/`      | https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset |

Each loader exposes the expected `path` in its `info` attribute. Adjust the file name as needed after downloading from Kaggle.

## How to run

```bash
python main.py                  # All datasets
python main.py fraud_detection  # A single dataset by info.name
```

## Team split

Each team member owns **one loader** (see TODOs in each file):
- fill `normalize()` with dataset-specific cleaning in `src/datasets/<name>.py`
- confirm/adjust `target_column` and `path`
- ensure correct column types
