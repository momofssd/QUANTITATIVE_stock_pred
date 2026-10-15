from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainingResult:
    model: Any
    model_type: str
    report: str
    feature_importance: pd.Series
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    validation_scores: dict[str, float]
    validation_baselines: dict[str, float]
    validation_details: dict[str, Any]


class SoftVoteEnsemble:
    def __init__(self, models: list[Pipeline], weights: list[float], model_names: list[str]):
        self.models = models
        self.weights = np.array(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.model_names = model_names
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        total = np.zeros((len(X), len(self.classes_)))
        for model, weight in zip(self.models, self.weights):
            model_proba = model.predict_proba(X)
            aligned = np.zeros_like(total)
            for idx, cls in enumerate(model.named_steps["classifier"].classes_):
                target_idx = int(np.where(self.classes_ == cls)[0][0])
                aligned[:, target_idx] = model_proba[:, idx]
            total += aligned * weight
        return total

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _candidate_classifiers(random_state: int = 42) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = [
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=160,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        ),
        (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=3,
                random_state=random_state,
            ),
        ),
        (
            "HistGradientBoostingClassifier",
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.04,
                max_leaf_nodes=18,
                random_state=random_state,
            ),
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        ),
    ]
    try:
        from xgboost import XGBClassifier

        candidates.insert(
            0,
            (
                "XGBClassifier",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        )
    except Exception:
        pass
    return candidates


def _build_classifier(model_name: str, random_state: int = 42):
    normalized = model_name.lower()
    if normalized in {"rf", "randomforest", "random_forest"}:
        return (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=160,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if normalized in {"extratrees", "extra_trees"}:
        return (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    if normalized in {"gradientboosting", "gb"}:
        return (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=3,
                random_state=random_state,
            ),
        )
    if normalized in {"histgradientboosting", "hgb"}:
        return (
            "HistGradientBoostingClassifier",
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.04,
                max_leaf_nodes=18,
                random_state=random_state,
            ),
        )
    if normalized in {"logistic", "logisticregression"}:
        return "LogisticRegression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if normalized in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBClassifier

            return "XGBClassifier", XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=1,
            )
        except Exception:
            pass
    return _build_classifier("randomforest", random_state)


def _fit_pipeline(classifier: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if isinstance(classifier, LogisticRegression):
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)
    return model


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "xgboost",
    test_fraction: float = 0.2,
    purge_gap: int = 0,
) -> TrainingResult:
    data = X.copy()
    labels = y.copy()
    valid = labels.notna()
    data = data.loc[valid]
    labels = labels.loc[valid].astype(int)
    valid = data.notna().sum(axis=1) > max(3, int(data.shape[1] * 0.5))
    data = data.loc[valid]
    labels = labels.loc[valid]
    if len(data) < 300:
        raise ValueError("Not enough historical rows after feature/label creation. Try an earlier start date.")

    split_idx = int(len(data) * (1 - test_fraction))
    train_end_idx = split_idx - max(purge_gap, 0)
    if train_end_idx < 1 or split_idx >= len(data):
        raise ValueError("Not enough rows for the requested validation split and purge gap.")
    X_train, X_test = data.iloc[:train_end_idx], data.iloc[split_idx:]
    y_train, y_test = labels.iloc[:train_end_idx], labels.iloc[split_idx:]

    use_ensemble = model_name.lower() in {"auto", "ensemble", "all"}
    validation_scores: dict[str, float] = {}
    if use_ensemble:
        fitted: list[tuple[str, Pipeline, float]] = []
        for candidate_name, classifier in _candidate_classifiers():
            candidate = _fit_pipeline(classifier, X_train, y_train)
            candidate_predictions = candidate.predict(X_test)
            score = f1_score(y_test, candidate_predictions, average="macro", zero_division=0)
            validation_scores[candidate_name] = float(score)
            fitted.append((candidate_name, candidate, max(score, 0.001)))
        fitted.sort(key=lambda item: item[2], reverse=True)
        selected = fitted[: min(4, len(fitted))]
        validation_model = SoftVoteEnsemble(
            models=[item[1] for item in selected],
            weights=[item[2] for item in selected],
            model_names=[item[0] for item in selected],
        )
        model_type = "WeightedSoftVoteEnsemble(" + ",".join(validation_model.model_names) + ")"
        refit_classifiers = dict(_candidate_classifiers())
        model = SoftVoteEnsemble(
            models=[_fit_pipeline(refit_classifiers[item[0]], data, labels) for item in selected],
            weights=[item[2] for item in selected],
            model_names=[item[0] for item in selected],
        )
    else:
        model_type, classifier = _build_classifier(model_name)
        validation_model = _fit_pipeline(classifier, X_train, y_train)
        predictions_for_score = validation_model.predict(X_test)
        validation_scores[model_type] = float(f1_score(y_test, predictions_for_score, average="macro", zero_division=0))
        _, final_classifier = _build_classifier(model_name)
        model = _fit_pipeline(final_classifier, data, labels)

    predictions = validation_model.predict(X_test)
    report = classification_report(y_test, predictions, labels=[0, 1, 2], target_names=["bearish", "sideways", "bullish"], zero_division=0)
    majority_label = int(y_train.mode().iloc[0])
    ma200_rule = np.where(X_test.get("close_above_200ma", pd.Series(0, index=X_test.index)) > 0, 2, 0)
    validation_baselines = {
        "AlwaysBullish": float(f1_score(y_test, np.full(len(y_test), 2), average="macro", zero_division=0)),
        "TrainingMajorityClass": float(f1_score(y_test, np.full(len(y_test), majority_label), average="macro", zero_division=0)),
        "MA200TrendRule": float(f1_score(y_test, ma200_rule, average="macro", zero_division=0)),
    }

    fitted_classifier = model.models[0].named_steps["classifier"] if isinstance(model, SoftVoteEnsemble) else model.named_steps["classifier"]
    if hasattr(fitted_classifier, "feature_importances_"):
        importance = pd.Series(fitted_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    else:
        importance = pd.Series(np.nan, index=X.columns)

    return TrainingResult(
        model=model,
        model_type=model_type,
        report=report,
        feature_importance=importance,
        train_start=str(X_train.index.min().date()),
        train_end=str(X_train.index.max().date()),
        test_start=str(X_test.index.min().date()),
        test_end=str(X_test.index.max().date()),
        validation_scores=validation_scores,
        validation_baselines=validation_baselines,
        validation_details={
            "validation_method": "purged_time_based_holdout",
            "purged_rows": max(purge_gap, 0),
            "final_fit_rows": len(data),
        },
    )


def predict_probabilities(model: Any, latest_features: pd.DataFrame) -> dict[str, float]:
    probabilities = model.predict_proba(latest_features)[0]
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["classifier"].classes_)
    output = {"bearish": 0.0, "sideways": 0.0, "bullish": 0.0}
    for cls, probability in zip(classes, probabilities):
        label = {0: "bearish", 1: "sideways", 2: "bullish"}.get(int(cls))
        if label:
            output[label] = float(probability)
    return output
