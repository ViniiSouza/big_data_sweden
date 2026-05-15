# Big Data Sweden — Halmstad Project

Big Data course project: compare a baseline (non-parallel) approach against Spark on datasets of different sizes for supervised classification.

## Structure

```
src/
  spark_session.py        # Shared SparkSession factory
  datasets/               # Spark (parallel) loaders
    base.py               # DatasetLoader (abstract) + DatasetInfo
    gen_z_social.py       # small/medium
    fraud_detection.py    # large
    lol_matches.py        # large
    job_salary.py         # small/medium
  datasets_pandas/        # Pandas (non-parallel) baseline — mirrors src/datasets/
    base.py               # DatasetLoader (abstract) + DatasetInfo, no pyspark imports
    gen_z_social.py
    fraud_detection.py
    lol_matches.py
    job_salary.py
main.py                   # Reads and normalizes every dataset (Spark or pandas)
data/raw/                 # CSVs downloaded from Kaggle (not versioned)
```

The `src/datasets_pandas/` package is intentionally free of any `pyspark` import — it should run on a machine without a working Spark/JDK setup.

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
python main.py                                # Spark engine (default), all datasets
python main.py fraud_detection                # Spark, a single dataset by info.name
python main.py --engine pandas                # Non-parallel pandas baseline, all
python main.py --engine pandas gen_z_social   # Pandas, a single dataset
```

The `--engine` flag selects between Spark (parallel) and pandas (non-parallel baseline). Both engines share the same dataset names, loader contract, and output shape, which makes them directly comparable for the project report.

## Team split

Each team member owns **one loader** (see TODOs in each file):
- fill `normalize()` with dataset-specific cleaning, on **both** sides (`src/datasets/<name>.py` for Spark and `src/datasets_pandas/<name>.py` for the pandas baseline)
- confirm/adjust `target_column` and `path`
- ensure correct column types
- keep the two implementations in sync: same final columns, same column order, equivalent type semantics (Spark `int` ↔ pandas `Int64`, Spark `double` ↔ pandas `float64`, Spark `string` ↔ pandas `string`)
