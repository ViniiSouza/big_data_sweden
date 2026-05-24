# Big Data Sweden — Halmstad Project

Big Data course project: supervised classification with Spark on datasets of different sizes.

## Structure

```
src/
  spark_session.py        # Shared SparkSession factory
  features.py             # build_features (StringIndexer + OHE + VectorAssembler)
  split.py                # train_test_split (fixed seed)
  train_spark.py          # train_and_evaluate (fit + transform + metrics)
  classifiers_spark.py    # CLASSIFIERS registry (random_forest, ...)
  experiments.py          # run_all loop: datasets x classifiers
  datasets/               # Spark loaders
    base.py               # DatasetLoader (abstract) + DatasetInfo
    gen_z_social.py       # small/medium
    fraud_detection.py    # large
    lol_matches.py        # large
    job_salary.py         # small/medium
main.py                   # CLI dispatcher: smoke | experiments
docs/adr/                 # Architecture Decision Records
CONTEXT.md                # Project glossary
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

The entry point is `main.py` with two subcommands: `smoke` and `experiments`. If no subcommand is given, `smoke` is the default.

### `smoke` — validate a dataset loader

Reads + normalizes the dataset(s) and prints schema + 5 sample rows. **No training.** Used while developing each loader's `normalize()`.

```bash
python main.py                          # all datasets (default subcommand)
python main.py smoke                    # explicit form, same as above
python main.py smoke fraud_detection    # single dataset by info.name
```

| Argument  | Type     | Default | Description |
|---|---|---|---|
| `dataset` | positional, optional | all | Run only the loader whose `info.name` matches (e.g. `gen_z_social`, `fraud_detection`, `lol_matches`, `job_salary`). |

### `experiments` — train and evaluate classifiers

Runs every `(dataset, classifier)` pair, prints metrics per experiment as they complete, writes a CSV, and prints a summary table + a per-classifier time breakdown + the grand wall-clock total at the end.

```bash
# Defaults: workers=1, partitions=1 (serial baseline), all datasets, all classifiers
python main.py experiments

# Filter to one dataset
python main.py experiments --dataset gen_z_social

# Filter to one classifier
python main.py experiments --classifier random_forest

# Parallelism sweep — increase cores and partitions to measure speedup
python main.py experiments --workers 4 --partitions 4
python main.py experiments --workers 8 --partitions 8

# Diagnostic: many cores but a single partition (shows the bottleneck)
python main.py experiments --workers 8 --partitions 1

# Combine — quick smoke of a single experiment
python main.py experiments --dataset gen_z_social --classifier naive_bayes --workers 4 --partitions 4
```

| Flag           | Type | Default | Description |
|---|---|---|---|
| `--dataset`    | str  | all     | Filter by dataset `info.name`. Values: `gen_z_social`, `fraud_detection`, `lol_matches`, `job_salary`. |
| `--classifier` | str  | all     | Filter by classifier name. Values: `random_forest`, `decision_tree`, `naive_bayes`. See [src/classifiers_spark.py](src/classifiers_spark.py). |
| `--workers`    | int  | `1`     | Spark local executor cores (`local[N]`). Default `1` is the fully serial baseline. |
| `--partitions` | int  | `1`     | Target partition count for the DataFrame after `load()`, plus the value of `spark.sql.shuffle.partitions`. Default `1` forces no parallelism. |

### Output

`experiments` produces three artifacts per run:

1. **CSV** at `results/experiments_w{workers}_p{partitions}.csv` — one row per `(dataset, classifier)` experiment. Columns: `dataset, classifier, workers, partitions, accuracy, f1, train_time_s, predict_time_s, total_time_s`. Rows are flushed incrementally — a crash mid-run keeps everything that completed.
2. **Stdout streaming** — one line per experiment as it finishes: `acc=... f1=... train=...s predict=...s total=...s`.
3. **Summary** at the end:
   - Table with every completed experiment.
   - Time spent per classifier (sum of `total_time_s` across datasets), sorted desc.
   - Grand total wall time (`time.perf_counter()` around the whole loop).

All times are in **seconds** (`time.perf_counter()`).

### Comparison sweep (recommended for the report)

```bash
python main.py experiments --workers 1 --partitions 1 \
 && python main.py experiments --workers 4 --partitions 4 \
 && python main.py experiments --workers 8 --partitions 8
```

Each run creates its own CSV (`experiments_w1_p1.csv`, `experiments_w4_p4.csv`, `experiments_w8_p8.csv`). To concatenate them for plotting:

```bash
cd results
{ head -1 experiments_w1_p1.csv; for f in experiments_w*.csv; do tail -n +2 "$f"; done; } > all_results.csv
```

### Classifiers

Registered in [src/classifiers_spark.py](src/classifiers_spark.py):

| Name             | Spark estimator              | Notes |
|---|---|---|
| `random_forest`  | `RandomForestClassifier`     | `numTrees=100, maxDepth=10, seed=42`. See [docs/adr/0001-rf-hyperparameters.md](docs/adr/0001-rf-hyperparameters.md). |
| `decision_tree`  | `DecisionTreeClassifier`     | `maxDepth=10, seed=42` — matches RF's depth so the comparison isolates the effect of bagging. |
| `naive_bayes`    | `NaiveBayes`                 | `modelType="gaussian"` — only mode that accepts the mixed numeric + OHE feature vector. |

Adding a new classifier is one function + one entry in the `CLASSIFIERS` dict; the runner picks it up automatically.

## Team split

Each team member owns **one loader** (see TODOs in each file):
- fill `normalize()` with dataset-specific cleaning in `src/datasets/<name>.py`
- confirm/adjust `target_column` and `path`
- ensure correct column types
