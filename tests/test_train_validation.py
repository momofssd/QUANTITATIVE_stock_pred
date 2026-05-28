import numpy as np
import pandas as pd

from src.train import train_classifier


def _training_frame(rows: int = 420) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    steps = np.arange(rows)
    cycle = np.sin(steps / 17)
    labels = pd.Series(1, index=index, name="label")
    labels[cycle > 0.45] = 2
    labels[cycle < -0.45] = 0
    features = pd.DataFrame(
        {
            "close_above_200ma": (cycle > -0.2).astype(int),
            "ma50_above_ma200": (cycle > 0.1).astype(int),
            "return_60d": cycle,
            "atr_pct": np.abs(np.cos(steps / 23)) / 20,
        },
        index=index,
    )
    return features, labels


def test_train_classifier_uses_purged_time_holdout():
    X, y = _training_frame()
    result = train_classifier(X, y, model_name="logistic", test_fraction=0.2, purge_gap=20)

    split_idx = int(len(X) * 0.8)
    expected_train_end_idx = split_idx - 20 - 1

    assert result.validation_details["validation_method"] == "purged_time_based_holdout"
    assert result.validation_details["purged_rows"] == 20
    assert result.train_end == str(X.index[expected_train_end_idx].date())
    assert result.test_start == str(X.index[split_idx].date())
    assert result.validation_details["final_fit_rows"] == len(X)
    assert "LogisticRegression" in result.validation_scores


def test_train_classifier_reports_baseline_scores():
    X, y = _training_frame()
    result = train_classifier(X, y, model_name="logistic", test_fraction=0.2, purge_gap=20)

    assert {"AlwaysBullish", "TrainingMajorityClass", "MA200TrendRule"}.issubset(result.validation_baselines)
    assert all(0 <= score <= 1 for score in result.validation_baselines.values())
