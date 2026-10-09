# Network Traffic Preprocessing and Class-Imbalance Study

A reproducible Python project that investigates how data preprocessing and class-imbalance techniques affect binary network intrusion detection, with a focus on attack recall, F1-score, balanced accuracy, PR-AUC, and false-alarm rates.

> **Research focus:** An IDS can report high accuracy while missing rare attacks. This project compares practical preprocessing and imbalance-handling strategies to make those trade-offs visible.

## Research questions

- How does feature standardization affect binary IDS performance?
- Does supervised feature selection retain useful attack-detection performance with fewer features?
- How do class weighting and SMOTE change attack recall, F1-score, and false-positive rate?
- How consistent are the results across stratified cross-validation folds?
- What computational cost accompanies each strategy?

## Methods

The experiment compares five strategies using a Logistic Regression classifier:

| Strategy | Method | Purpose |
|---|---|---|
| `baseline` | Logistic Regression without scaling | Establish a reference result |
| `standardized` | `StandardScaler` + Logistic Regression | Put numeric features on comparable scales |
| `feature_selection` | Scaling + mutual-information `SelectKBest` + Logistic Regression | Assess the effect of reducing the feature set |
| `class_weighted` | Scaling + balanced class weights + Logistic Regression | Increase the learning penalty for minority-class errors |
| `smote` | Scaling + SMOTE + Logistic Regression | Oversample the minority class in training data |

Preprocessing and feature selection are implemented in scikit-learn pipelines, so transformations are fitted within the relevant training folds. SMOTE is applied to training data only, never to the held-out test set.

## Example results

The following is a **single preliminary run** using the configuration shown below, not a claim of universal performance. The loader found 9,625,148 rows across 14 CSV files, sampled up to 100,000 rows, and retained 90,250 usable rows after cleaning and duplicate removal. The resulting dataset had 78 numeric features.

| Strategy | Accuracy | Balanced accuracy | Precision | Attack recall | F1 | PR-AUC | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standardized | 0.9458 | **0.9383** | **0.8622** | 0.9238 | **0.8920** | **0.9125** | 0.9628 | **0.0471** | 0.0762 |
| SMOTE | 0.9347 | 0.9354 | 0.8193 | 0.9368 | 0.8741 | 0.8977 | 0.9637 | 0.0660 | 0.0632 |
| Class weighted | 0.9346 | 0.9355 | 0.8190 | **0.9371** | 0.8741 | 0.9015 | **0.9640** | 0.0662 | **0.0629** |
| Feature selection | 0.8700 | 0.8154 | 0.7423 | 0.7094 | 0.7255 | 0.7788 | 0.9176 | 0.0787 | 0.2906 |
| Baseline | 0.8485 | 0.7579 | 0.7368 | 0.5822 | 0.6504 | 0.7032 | 0.8828 | 0.0664 | 0.4178 |

### Preliminary interpretation

- **Standardization** achieved the strongest F1-score and balanced accuracy in this run, with the lowest false-positive rate.
- **Class weighting** achieved the highest attack recall, while also producing a higher false-positive rate than standardization.
- **SMOTE** produced a similar recall/false-alarm trade-off to class weighting in this run.
- **Feature selection** with the current top-20 setting reduced performance substantially; alternative feature counts and selectors should be tested before drawing a final conclusion.

These results are specific to this sampled dataset, split, preprocessing configuration, and random seed. They should be reproduced across multiple seeds and, ideally, validated on a separate traffic capture before being used to support a research claim.

> **Fair-comparison note:** In the current configuration, the SMOTE strategy is capped at 50,000 training rows (`smote_max_rows: 50000`), while the other strategies use the full training split. Runtime and performance comparisons are therefore not perfectly matched. For a stricter comparison, align training sample sizes or document this limitation clearly.

## Metrics

The results table includes:

- Accuracy and balanced accuracy
- Precision, recall / minority-attack recall, and F1-score
- Precision–Recall AUC (average precision) and ROC-AUC
- False-positive rate (FPR) and false-negative rate (FNR)
- Cross-validation F1 mean and standard deviation
- Cross-validation balanced accuracy mean
- Elapsed cross-validation time (`fit_seconds` in the CSV)

For imbalanced IDS data, do not rely on accuracy alone. Review attack recall, F1, PR-AUC, balanced accuracy, and false-positive rate together.

## Dataset requirements

This repository does **not** include network traffic datasets. Keep dataset files locally and follow the dataset provider's terms and citation requirements.

The project accepts one CSV file or multiple CSV files in `data/raw/`. Each file should contain:

- A target column named `Label` by default (configurable in `configs/config.yaml`).
- Numeric network-flow features shared across the CSV files.
- Both benign/normal traffic and attack traffic across the combined input files.

For labels such as `BENIGN` or `Normal`, the loader treats benign/normal rows as class 0 and other labels as attack class 1. If no recognizable benign label is present, exactly two distinct labels are required. This project evaluates **binary normal-versus-attack classification**; it does not currently produce per-attack-family multiclass scores.

### Add your data

Place all relevant CSV files—including benign/normal traffic—in:

```text
data/raw/
├── benign_or_normal.csv
├── attack_file_1.csv
├── attack_file_2.csv
└── ...
```

The filenames above are examples. Keep the original dataset filenames if desired. The CSVs must have compatible column headers and the target column in every file.

### Dataset-size controls

Default settings are designed to make an initial run more manageable on a personal computer:

```yaml
data_path: data/raw
target_column: Label
max_rows: 100000
smote_max_rows: 50000
```

`max_rows` is a **global cap across all CSV files**, not a cap per file. The loader samples proportionally to source-file row counts, then cleans missing/infinite values and duplicate rows. As a result, the number of usable rows may be lower than `max_rows`.

After confirming that the pipeline works and checking available memory, set `max_rows: null` to load all rows. Larger runs may need substantially more RAM and time.

## Getting started in PyCharm

### 1. Open the project

Extract or clone this repository and open the project root in PyCharm—the folder containing `run_project.py`, `requirements.txt`, and `configs/`.

### 2. Create/select a Python environment

Use Python 3.10 or newer. In PyCharm, configure the project's virtual environment as the interpreter.

### 3. Install dependencies

From the project root terminal, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place the CSV file(s) in `data/raw/`. Ensure the combined data includes benign/normal and attack traffic.

### 5. Check the configuration

Edit `configs/config.yaml` if needed:

```yaml
data_path: data/raw
target_column: Label
output_dir: outputs
random_state: 42
test_size: 0.20
cv_folds: 5
max_rows: 100000
smote_max_rows: 50000
top_k_features: 20
n_jobs: -1
strategies:
  - baseline
  - standardized
  - feature_selection
  - class_weighted
  - smote
```

### 6. Run the entire project

Run **`run_project.py`** in PyCharm, or from a terminal at the project root:

```bash
python run_project.py
```

This is the dedicated entry point for the whole experiment. Avoid running `network_imbalance/experiment.py` directly because it uses package-relative imports.

## Outputs

After a successful run, the project writes:

```text
outputs/
├── figures/
│   ├── class_distribution.png
│   └── strategy_comparison.png
└── tables/
    ├── class_distribution.csv
    └── experiment_results.csv
```

- `class_distribution.csv`: class counts and proportions.
- `experiment_results.csv`: metrics for each strategy.
- `class_distribution.png`: class-distribution plot.
- `strategy_comparison.png`: F1, attack recall, and balanced-accuracy comparison.

The `outputs/` directory is ignored by Git by default to keep the repository clean. Export selected tables/figures separately if you want to publish a particular reproducible result, including the configuration and dataset-sampling details.

## Repository layout

```text
network-traffic-preprocessing-imbalance/
├── run_project.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   └── raw/                 # local dataset files; not committed
├── network_imbalance/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   └── experiment.py
├── tests/
│   └── test_data.py
└── outputs/                 # generated results; ignored by Git
```

## Reproducibility and research limitations

- The default random seed is `42`; change it in the configuration for robustness checks.
- The current workflow uses a stratified 80/20 train/test split and five-fold stratified cross-validation on the training portion.
- The current study is binary and uses Logistic Regression for all strategies to focus the comparison on preprocessing and imbalance handling.
- The current dataset cap may change class proportions compared with the full dataset; report the sampled and cleaned class counts.
- The current cross-validation runtime field is elapsed time around `cross_validate`, not just the duration of one model fit.
- The current implementation does not yet include paired significance tests, repeated-seed confidence intervals, multiclass attack-family analysis, or cross-dataset validation.
- A random row split can place very similar flows from the same capture period in both train and test sets. For stronger generalization evidence, consider a split by capture day, source file, or time period.

Recommended next steps include repeating experiments over multiple seeds, adding paired statistical comparisons, evaluating more feature counts, matching sample sizes across strategies, testing on another IDS dataset, and measuring inference latency.

## GitHub topics

Suggested repository topics: `intrusion-detection`, `network-security`, `class-imbalance`, `smote`, `feature-selection`, `machine-learning`, `scikit-learn`, `cybersecurity`, `data-preprocessing`, `python`.

## References

- [scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [scikit-learn: SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
- [imbalanced-learn: SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

Cite the exact network intrusion dataset you used according to its official documentation or publication. This repository does not redistribute the data.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
