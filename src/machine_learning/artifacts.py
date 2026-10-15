"""Save fitted estimators separately from quantitative report generation."""

from datetime import date
from pathlib import Path

import joblib

from ..utils import MODEL_DIR
from .five_day_forecast import FiveDayTrainingResult
from .train import TrainingResult


def save_trained_models(
    ticker: str,
    training: TrainingResult,
    five_day_training: FiveDayTrainingResult,
    model_dir: Path = MODEL_DIR,
) -> tuple[Path, Path]:
    """Write long-horizon and five-session joblib artifacts; return their paths.

    Filenames use today's date and overwrite an existing same-day artifact.
    The five-session file includes calibrated models, feature order, interval
    calibration, and the frozen historical validation report.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    date_part = date.today().isoformat()
    model_path = model_dir / f"{ticker.lower()}_{training.model_type.lower()}_{date_part}.joblib"
    five_day_model_path = model_dir / f"{ticker.lower()}_five_day_forecast_{date_part}.joblib"
    joblib.dump(training.model, model_path)
    joblib.dump(
        {
            "direction_model": five_day_training.direction_model,
            "return_model": five_day_training.return_model,
            "conditional_return_models": five_day_training.conditional_return_models,
            "feature_columns": five_day_training.feature_columns,
            "interval_radius": five_day_training.interval_radius,
            "validation": five_day_training.validation,
            "live_fit": five_day_training.live_fit,
            "calibration_residuals": five_day_training.calibration_residuals,
            "session_models": five_day_training.session_models,
        },
        five_day_model_path,
    )
    return model_path, five_day_model_path
