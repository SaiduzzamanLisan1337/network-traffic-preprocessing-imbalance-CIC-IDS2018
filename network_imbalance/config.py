from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    data_path: str = "data/raw"
    target_column: str = "Label"
    output_dir: str = "outputs"
    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    max_rows: int | None = 100000
    smote_max_rows: int = 50000
    top_k_features: int = 20
    n_jobs: int = -1
    strategies: list[str] = field(default_factory=lambda: [
        "baseline", "standardized", "feature_selection",
        "class_weighted", "smote"
    ])
