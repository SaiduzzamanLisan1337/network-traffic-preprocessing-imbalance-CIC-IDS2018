"""Load one IDS CSV or combine multiple CSV files from a directory."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


def _clean_column_name(name):
    """Remove common BOM/whitespace issues in CICIDS-style CSV headers."""
    return str(name).replace("\ufeff", "").strip()


def _read_header(csv_path, target_column):
    columns = [_clean_column_name(c) for c in pd.read_csv(
        csv_path, nrows=0, encoding="utf-8-sig"
    ).columns]
    target_matches = [c for c in columns if c.casefold() == target_column.casefold()]
    if not target_matches:
        raise ValueError(
            f"Target column '{target_column}' was not found in {csv_path.name}. "
            f"Available columns include: {columns[:15]}"
        )
    actual_target = target_matches[0]
    return [target_column if c == actual_target else c for c in columns]


def _count_data_rows(csv_path):
    """Fast row count for typical network-flow CSVs (one record per line)."""
    with csv_path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)  # exclude header


def _allocate_quotas(row_counts, cap):
    """Allocate a global row cap in proportion to the source file sizes."""
    total = sum(row_counts)
    if total <= cap:
        return list(row_counts)
    active = [i for i, count in enumerate(row_counts) if count > 0]
    if cap < len(active):
        raise ValueError(
            f"max_rows={cap} is too small to sample from {len(active)} non-empty CSV files. "
            "Increase max_rows or set it to null."
        )

    ideal = [cap * count / total for count in row_counts]
    quotas = [min(count, int(value)) for count, value in zip(row_counts, ideal)]
    for i in active:
        if quotas[i] == 0:
            quotas[i] = 1

    while sum(quotas) > cap:
        candidates = [i for i in active if quotas[i] > 1]
        if not candidates:
            break
        i = max(candidates, key=lambda j: quotas[j] - ideal[j])
        quotas[i] -= 1

    while sum(quotas) < cap:
        candidates = [i for i in active if quotas[i] < row_counts[i]]
        if not candidates:
            break
        i = max(candidates, key=lambda j: ideal[j] - quotas[j])
        quotas[i] += 1
    return quotas


def _read_sample(csv_path, target_column, quota, row_count, rng):
    """Read a random row sample without loading an entire large CSV into RAM."""
    if quota <= 0 or row_count <= 0:
        return pd.DataFrame()
    kwargs = {"encoding": "utf-8-sig", "low_memory": False}
    if quota < row_count:
        chosen = set(rng.choice(row_count, size=quota, replace=False).tolist())
        # pandas counts the header as physical line 0; data row 0 is line 1.
        kwargs["skiprows"] = lambda line_num: (
            line_num > 0 and (line_num - 1) not in chosen
        )
    df = pd.read_csv(csv_path, **kwargs)
    df.columns = [_clean_column_name(c) for c in df.columns]
    target_matches = [c for c in df.columns if c.casefold() == target_column.casefold()]
    if not target_matches:
        raise ValueError(f"Target column '{target_column}' was not found in {csv_path.name}.")
    actual_target = target_matches[0]
    if actual_target != target_column:
        df = df.rename(columns={actual_target: target_column})
    return df


def load_dataset(path, target_column, max_rows=None, random_state=42):
    """Load a single CSV or combine every ``*.csv`` in a directory.

    When ``max_rows`` is set, it is a global cap across all source CSVs. Rows are
    randomly sampled in proportion to each file's row count to avoid loading the
    full dataset into memory and to preserve approximate source-class proportions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {path}. Put your CSV file(s) in data/raw/ "
            "or update data_path in configs/config.yaml."
        )

    if path.is_dir():
        csv_files = sorted(p for p in path.glob("*.csv") if p.is_file())
        if not csv_files:
            raise FileNotFoundError(
                f"No .csv files found in {path}. Place your IDS CSV files in this folder."
            )
    elif path.is_file() and path.suffix.lower() == ".csv":
        csv_files = [path]
    else:
        raise ValueError(f"Dataset path must be a CSV file or directory of CSV files: {path}")

    # Check headers first so files with inconsistent features can be handled safely.
    headers = [_read_header(csv_file, target_column) for csv_file in csv_files]
    common = set(headers[0])
    for header in headers[1:]:
        common.intersection_update(header)
    common_columns = [c for c in headers[0] if c in common]
    if target_column not in common_columns:
        raise ValueError(
            f"Target column '{target_column}' must be present in every CSV file in {path}."
        )
    feature_count = len([c for c in common_columns if c != target_column])
    if feature_count == 0:
        raise ValueError("No common feature columns were found across the input CSV files.")
    if any(set(header) != set(headers[0]) for header in headers[1:]):
        warnings.warn(
            "The CSV files do not have identical columns. Only columns shared by every file "
            "will be used, to avoid introducing missing values during concatenation.",
            UserWarning,
        )

    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError("max_rows must be a positive integer or null.")

    row_counts = [_count_data_rows(csv_file) for csv_file in csv_files]
    total_rows = sum(row_counts)
    if total_rows == 0:
        raise ValueError("The selected CSV file(s) contain no data rows.")

    if max_rows is None or total_rows <= int(max_rows):
        quotas = row_counts
    else:
        quotas = _allocate_quotas(row_counts, int(max_rows))
        print(
            f"Sampling {sum(quotas):,} rows from {total_rows:,} available rows "
            f"across {len(csv_files)} CSV files (max_rows={int(max_rows):,})."
        )

    rng = np.random.default_rng(random_state)
    frames = []
    for csv_file, quota, count in zip(csv_files, quotas, row_counts):
        if quota == 0:
            continue
        df = _read_sample(csv_file, target_column, quota, count, rng)
        df = df.reindex(columns=common_columns)
        frames.append(df)

    if not frames:
        raise ValueError("No rows could be loaded from the selected CSV files.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    combined = combined.drop_duplicates()
    y = combined[target_column].astype(str).str.strip()
    X = combined.drop(columns=[target_column]).select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after preprocessing.")
    if y.nunique() < 2:
        raise ValueError(
            "The loaded data contains fewer than two label classes after cleaning. "
            "For an IDS imbalance study, include both benign/normal traffic and attack traffic."
        )
    print(f"Loaded {len(csv_files)} CSV file(s); {len(X):,} usable rows, {X.shape[1]} numeric features.")
    return X, y


def class_distribution(y):
    counts = y.value_counts()
    return counts.to_frame("count").assign(
        proportion=lambda d: d["count"] / d["count"].sum()
    )
