"""Deterministic features, training targets, decision rules, and report shaping.

This layer accepts market frames and model outputs; it does not fetch data,
fit estimators, or call an AI provider.
"""

from .decision_engine import DecisionResult, calculate_decision
from .features import FEATURE_COLUMNS, MACRO_FEATURE_COLUMNS, build_features, model_matrix, short_term_model_matrix
from .short_term_features import SHORT_TERM_FEATURE_COLUMNS, build_short_term_features
from .json_exporter import build_signal_json, save_signal_outputs
from .labels import create_basic_labels, create_direction_labels, create_forward_return_target, infer_regime
from .short_term_trade import build_short_term_trade_plan

__all__ = [
    "DecisionResult", "FEATURE_COLUMNS", "MACRO_FEATURE_COLUMNS", "build_features",
    "model_matrix", "short_term_model_matrix", "SHORT_TERM_FEATURE_COLUMNS", "build_short_term_features",
    "create_basic_labels", "create_direction_labels",
    "create_forward_return_target", "infer_regime", "calculate_decision",
    "build_short_term_trade_plan", "build_signal_json", "save_signal_outputs",
]
