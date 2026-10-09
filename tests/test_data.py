import pandas as pd

from network_imbalance.data import class_distribution, load_dataset


def test_class_distribution():
    y = pd.Series(["BENIGN", "BENIGN", "ATTACK"])
    d = class_distribution(y)
    assert d.loc["BENIGN", "count"] == 2
    assert abs(d["proportion"].sum() - 1) < 1e-9


def test_load_combines_csv_files_from_directory(tmp_path):
    (tmp_path / "benign.csv").write_text(
        "feature_a,feature_b,Label\n1,10,BENIGN\n2,20,BENIGN\n", encoding="utf-8"
    )
    (tmp_path / "attack.csv").write_text(
        "feature_a,feature_b,Label\n9,90,DoS\n8,80,DoS\n", encoding="utf-8"
    )

    X, y = load_dataset(tmp_path, "Label", max_rows=None)

    assert len(X) == 4
    assert list(X.columns) == ["feature_a", "feature_b"]
    assert set(y) == {"BENIGN", "DoS"}
