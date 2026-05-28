from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from .features import MACRO_FEATURE_COLUMNS
from .labels import create_direction_labels, create_forward_return_target
from .utils import clean_float, confidence_label


FIVE_DAY_HORIZON = 5
TREE_ESTIMATORS = 100


@dataclass
class FiveDayTrainingResult:
    direction_model: Any
    return_model: Any
    conditional_return_models: dict[str, Any]
    direction_validation_macro_f1: float
    return_validation_mae: float
    conditional_return_validation_mae: dict[str, float]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    training_rows: int


def _direction_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=TREE_ESTIMATORS,
                    max_depth=8,
                    min_samples_leaf=6,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _return_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=TREE_ESTIMATORS,
                    max_depth=8,
                    min_samples_leaf=6,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def train_five_day_models(
    X: pd.DataFrame,
    features: pd.DataFrame,
    horizon_days: int = FIVE_DAY_HORIZON,
    test_fraction: float = 0.2,
) -> FiveDayTrainingResult:
    labels = create_direction_labels(features, horizon_days=horizon_days)
    returns = create_forward_return_target(features, horizon_days=horizon_days)
    valid = labels.notna() & returns.notna()
    data = X.loc[valid].copy()
    labels = labels.loc[valid].astype(int)
    returns = returns.loc[valid].astype(float)
    usable = data.notna().sum(axis=1) > max(3, int(data.shape[1] * 0.5))
    data = data.loc[usable]
    labels = labels.loc[usable]
    returns = returns.loc[usable]
    if len(data) < 120:
        raise ValueError("Not enough historical rows for the five-trading-day forecast.")

    split_idx = int(len(data) * (1 - test_fraction))
    train_end_idx = split_idx - horizon_days
    if train_end_idx < 1 or split_idx >= len(data):
        raise ValueError("Not enough rows for five-trading-day validation.")
    X_train, X_test = data.iloc[:train_end_idx], data.iloc[split_idx:]
    y_train, y_test = labels.iloc[:train_end_idx], labels.iloc[split_idx:]
    return_train, return_test = returns.iloc[:train_end_idx], returns.iloc[split_idx:]

    validation_direction_model = _direction_pipeline().fit(X_train, y_train)
    validation_return_model = _return_pipeline().fit(X_train, return_train)
    direction_score = f1_score(y_test, validation_direction_model.predict(X_test), average="macro", zero_division=0)
    return_mae = mean_absolute_error(return_test, validation_return_model.predict(X_test))

    direction_model = _direction_pipeline().fit(data, labels)
    return_model = _return_pipeline().fit(data, returns)
    conditional_return_models: dict[str, Any] = {}
    conditional_return_validation_mae: dict[str, float] = {}
    for cls, name in [(0, "bearish"), (1, "bullish")]:
        train_mask = y_train == cls
        full_mask = labels == cls
        test_mask = y_test == cls
        if train_mask.sum() < 20 or full_mask.sum() < 20:
            continue
        validation_model = _return_pipeline().fit(X_train.loc[train_mask], return_train.loc[train_mask])
        conditional_return_validation_mae[name] = (
            float(mean_absolute_error(return_test.loc[test_mask], validation_model.predict(X_test.loc[test_mask])))
            if test_mask.any()
            else float(return_mae)
        )
        conditional_return_models[name] = _return_pipeline().fit(data.loc[full_mask], returns.loc[full_mask])
    return FiveDayTrainingResult(
        direction_model=direction_model,
        return_model=return_model,
        conditional_return_models=conditional_return_models,
        direction_validation_macro_f1=float(direction_score),
        return_validation_mae=float(return_mae),
        conditional_return_validation_mae=conditional_return_validation_mae,
        train_start=str(X_train.index.min().date()),
        train_end=str(X_train.index.max().date()),
        test_start=str(X_test.index.min().date()),
        test_end=str(X_test.index.max().date()),
        training_rows=len(data),
    )


def generate_five_day_forecast(
    X: pd.DataFrame,
    features: pd.DataFrame,
    latest_index: pd.Timestamp,
    horizon_days: int = FIVE_DAY_HORIZON,
) -> tuple[dict, FiveDayTrainingResult]:
    training = train_five_day_models(X, features, horizon_days=horizon_days)
    latest_X = X.loc[[latest_index]]
    class_probabilities = training.direction_model.predict_proba(latest_X)[0]
    classes = training.direction_model.named_steps["classifier"].classes_
    probabilities = {"bearish": 0.0, "bullish": 0.0}
    for cls, probability in zip(classes, class_probabilities):
        probabilities["bullish" if int(cls) == 1 else "bearish"] = float(probability)

    bullish = probabilities["bullish"]
    bearish = probabilities["bearish"]
    direction = "BULLISH" if bullish >= bearish else "BEARISH"
    direction_key = direction.lower()
    confidence = max(bullish, bearish)
    current_price = float(features.loc[latest_index, "close"])
    unconditional_return = max(float(training.return_model.predict(latest_X)[0]), -0.99)
    selected_return_model = training.conditional_return_models.get(direction_key, training.return_model)
    projected_return = max(float(selected_return_model.predict(latest_X)[0]), -0.99)
    possible_price = current_price * (1 + projected_return)
    unconditional_price = current_price * (1 + unconditional_return)
    price_projection_direction = "BULLISH" if projected_return >= 0 else "BEARISH"
    error_band = training.conditional_return_validation_mae.get(direction_key, training.return_validation_mae)

    forecast = {
        "horizon_trading_days": horizon_days,
        "predicted_direction": direction,
        "confidence_score": clean_float(confidence),
        "confidence_label": confidence_label(confidence),
        "confidence_basis": "direction_model_probability",
        "direction_probabilities": {
            "bullish": clean_float(bullish),
            "bearish": clean_float(bearish),
        },
        "current_price": clean_float(current_price, 2),
        "projected_return": clean_float(projected_return),
        "possible_price_in_5_trading_days": clean_float(possible_price, 2),
        "price_estimate_basis": "direction_conditioned_forward_return_regression",
        "unconditional_expected_return": clean_float(unconditional_return),
        "unconditional_expected_price_in_5_trading_days": clean_float(unconditional_price, 2),
        "price_projection_direction": price_projection_direction,
        "direction_price_models_agree": direction == price_projection_direction,
        "possible_price_range": {
            "low": clean_float(max(current_price * (1 + projected_return - error_band), 0), 2),
            "high": clean_float(current_price * (1 + projected_return + error_band), 2),
            "basis": "validation_mean_absolute_return_error",
        },
        "validation": {
            "method": "purged_time_based_holdout",
            "purged_rows": horizon_days,
            "direction_macro_f1": clean_float(training.direction_validation_macro_f1),
            "return_mean_absolute_error": clean_float(training.return_validation_mae),
            "conditional_return_mean_absolute_error": {
                name: clean_float(value) for name, value in training.conditional_return_validation_mae.items()
            },
            "train_start": training.train_start,
            "train_end": training.train_end,
            "test_start": training.test_start,
            "test_end": training.test_end,
        },
        "model_metadata": {
            "direction_model_type": "ExtraTreesClassifier",
            "price_model_type": "ExtraTreesRegressor",
            "price_model_conditioning": "predicted_direction",
            "training_rows": training.training_rows,
            "feature_count": len(X.columns),
            "macro_feature_columns": [name for name in MACRO_FEATURE_COLUMNS if name in X.columns],
        },
        "interpretation_note": (
            "The direction classifier and direction-conditioned price model disagree; treat this as an uncertain "
            "five-session forecast."
            if direction != price_projection_direction
            else "The possible price is estimated from historical returns conditioned on the predicted direction."
        ),
    }
    return forecast, training
