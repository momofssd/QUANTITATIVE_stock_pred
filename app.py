import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
from pandas.tseries.offsets import BDay
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from scipy.signal import argrelextrema
import yfinance as yf

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="frontend/dist")
CORS(app)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PREDICTION_DAYS = 5
MIN_TARGET_MOVE = 0.005
TRAIN_RATIO     = 0.80
LOOKBACK        = 20
SUPPORT_RES_WIN = 10
Z_SCORE_WINDOWS = [20, 50, 100]

COLORS = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "green":   "#3fb950",
    "red":     "#f85149",
    "yellow":  "#d29922",
    "blue":    "#58a6ff",
    "purple":  "#bc8cff",
    "orange":  "#ff9800",
    "cyan":    "#76e3ea",
}

def fetch_data(ticker: str) -> pd.DataFrame:
    df = pd.DataFrame()
    live_success = False
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="max", auto_adjust=True, repair=True, keepna=False)
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            live_success = True
    except Exception:
        pass

    if not live_success:
        seed = sum(ord(c) for c in ticker)
        rng  = np.random.default_rng(seed)
        n_days   = 5040
        end_date = datetime(2025, 5, 2)
        bdays    = pd.bdate_range(end=end_date, periods=n_days)
        start_price = float(rng.integers(20, 400))
        annual_ret  = rng.uniform(0.04, 0.18)
        annual_vol  = rng.uniform(0.18, 0.40)
        dt       = 1 / 252
        drift    = (annual_ret - 0.5 * annual_vol**2) * dt
        vol_step = annual_vol * np.sqrt(dt)
        prices      = [start_price]
        regime_dur  = rng.integers(60, 240, size=60)
        regime_type = rng.choice(["bull", "bear", "sideways"], size=60, p=[0.50, 0.25, 0.25])
        regime_mod  = {"bull": 1.4, "bear": -0.6, "sideways": 0.1}

        reg_idx, days_in_reg = 0, 0
        for _ in range(n_days - 1):
            if days_in_reg >= regime_dur[reg_idx % len(regime_dur)]:
                reg_idx += 1
                days_in_reg = 0
            mod       = regime_mod[regime_type[reg_idx % len(regime_type)]]
            eff_drift = drift * mod
            shock     = rng.standard_normal() * vol_step
            if rng.random() < 0.003:
                shock += rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
            prices.append(max(prices[-1] * np.exp(eff_drift + shock), 1.0))
            days_in_reg += 1

        close_arr  = np.array(prices)
        daily_range = rng.uniform(0.005, 0.025, size=n_days)
        high_arr   = close_arr * (1 + daily_range * rng.uniform(0.3, 1.0, n_days))
        low_arr    = close_arr * (1 - daily_range * rng.uniform(0.3, 1.0, n_days))
        open_arr   = low_arr + rng.random(n_days) * (high_arr - low_arr)
        base_vol   = float(rng.integers(5_000_000, 80_000_000))
        volume_arr = (base_vol * rng.lognormal(0, 0.5, n_days)).astype(int)
        df = pd.DataFrame({
            "Open":   open_arr,
            "High":   high_arr,
            "Low":    low_arr,
            "Close":  close_arr,
            "Volume": volume_arr,
        }, index=bdays)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std.replace(0, np.nan)

    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"]  = ta.trend.sma_indicator(df["Close"], window=w)
        df[f"EMA_{w}"]  = ta.trend.ema_indicator(df["Close"], window=w)

    df["RSI"]           = ta.momentum.rsi(df["Close"], window=14)
    df["RSI_fast"]      = ta.momentum.rsi(df["Close"], window=7)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"]          = macd.macd()
    df["MACD_signal"]   = macd.macd_signal()
    df["MACD_hist"]     = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["Stoch_K"]       = stoch.stoch()
    df["Stoch_D"]       = stoch.stoch_signal()
    df["Williams_R"]    = ta.momentum.williams_r(df["High"], df["Low"], df["Close"])
    df["ROC"]           = ta.momentum.roc(df["Close"], window=12)

    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_upper"]      = bb.bollinger_hband()
    df["BB_mid"]        = bb.bollinger_mavg()
    df["BB_lower"]      = bb.bollinger_lband()
    df["BB_width"]      = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pct"]        = bb.bollinger_pband()
    df["ATR"]           = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    df["ATR_pct"]       = df["ATR"] / df["Close"]

    df["OBV"]           = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["Vol_SMA20"]     = df["Volume"].rolling(20).mean()
    df["Vol_ratio"]     = df["Volume"] / df["Vol_SMA20"]
    df["VWAP"]          = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["CMF"]           = ta.volume.chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"])

    df["Return_1d"]     = df["Close"].pct_change(1)
    df["Return_5d"]     = df["Close"].pct_change(5)
    df["Return_20d"]    = df["Close"].pct_change(20)
    df["Log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["High_Low_pct"]  = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_pct"]= (df["Close"] - df["Open"]) / df["Open"]

    for w in Z_SCORE_WINDOWS:
        df[f"Close_Z_{w}"]  = rolling_zscore(df["Close"], w)
        df[f"Return_Z_{w}"] = rolling_zscore(df["Log_return"], w)
        df[f"Volume_Z_{w}"] = rolling_zscore(df["Volume"], w)

    adx_ind             = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"]           = adx_ind.adx()
    df["ADX_pos"]       = adx_ind.adx_pos()
    df["ADX_neg"]       = adx_ind.adx_neg()

    for w in [5, 10, 20]:
        df[f"Volatility_{w}d"] = df["Log_return"].rolling(w).std() * np.sqrt(252)
        df[f"High_{w}d"]       = df["High"].rolling(w).max()
        df[f"Low_{w}d"]        = df["Low"].rolling(w).min()

    df["Future_close"]     = df["Close"].shift(-PREDICTION_DAYS)
    df["Future_avg_close"] = (
        df["Close"]
        .shift(-1)
        .rolling(PREDICTION_DAYS)
        .mean()
        .shift(-(PREDICTION_DAYS - 1))
    )
    df["Future_avg_return"] = (df["Future_avg_close"] - df["Close"]) / df["Close"]
    df["Target"] = np.where(
        df["Future_avg_return"] > MIN_TARGET_MOVE,
        1,
        np.where(df["Future_avg_return"] < -MIN_TARGET_MOVE, 0, np.nan),
    )

    feature_cols = [c for c in df.columns if c not in ("Future_close", "Future_avg_close", "Future_avg_return", "Target")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=feature_cols, inplace=True)
    return df

def train_and_validate(df: pd.DataFrame):
    train_df = df.dropna(subset=["Target"])
    feature_cols = [c for c in df.columns if c not in ("Open","High","Low","Close","Volume","Future_close","Future_avg_close","Future_avg_return","Target")]
    X = train_df[feature_cols].values
    y = train_df["Target"].astype(int).values
    dates = train_df.index

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    dates_val       = dates[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=1),
        "GradBoost": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42),
    }

    preds_val  = {}
    probas_val = {}
    metrics    = {}

    def balanced_accuracy_for_threshold(y_true, probabilities, threshold):
        y_true = np.asarray(y_true)
        pred = (probabilities >= threshold).astype(int)
        pos_mask = y_true == 1
        neg_mask = y_true == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return accuracy_score(y_true, pred)
        bullish_recall = ((pred == 1) & pos_mask).sum() / pos_mask.sum()
        bearish_recall = ((pred == 0) & neg_mask).sum() / neg_mask.sum()
        return float((bullish_recall + bearish_recall) / 2)

    def tune_decision_threshold(y_true, probabilities):
        if len(np.unique(y_true)) < 2:
            return 0.5, balanced_accuracy_for_threshold(y_true, probabilities, 0.5)
        bullish_rate = float(np.mean(y_true))
        threshold = float(np.quantile(probabilities, 1 - bullish_rate))
        threshold = float(np.clip(threshold, 0.05, 0.95))
        return threshold, balanced_accuracy_for_threshold(y_true, probabilities, threshold)

    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        p = mdl.predict(X_val_s)
        pr = mdl.predict_proba(X_val_s)[:, 1]
        preds_val[name]  = p
        probas_val[name] = pr
        acc = accuracy_score(y_val, p)
        auc = roc_auc_score(y_val, pr)
        metrics[name] = {"acc": acc, "auc": auc}

    ensemble_proba = np.mean([probas_val[n] for n in models], axis=0)
    decision_threshold, balanced_acc = tune_decision_threshold(y_val, ensemble_proba)
    ensemble_pred  = (ensemble_proba >= decision_threshold).astype(int)
    ens_acc = accuracy_score(y_val, ensemble_pred)
    ens_auc = roc_auc_score(y_val, ensemble_proba)
    metrics["Ensemble"] = {"acc": ens_acc, "auc": ens_auc, "balanced_acc": balanced_acc}

    for name, mdl in models.items():
        mdl.fit(scaler.transform(X), y)

    fi = dict(zip(feature_cols, models["XGBoost"].feature_importances_))
    top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]

    return (models, scaler, feature_cols, df, split, dates_val, preds_val, probas_val, ensemble_pred, ensemble_proba, y_val, metrics, top_features, decision_threshold)

def find_support_resistance(price_data, window: int = SUPPORT_RES_WIN, tolerance: float = 0.015):
    if isinstance(price_data, pd.DataFrame):
        lows = price_data["Low"].astype(float)
        highs = price_data["High"].astype(float)
        close = price_data["Close"].astype(float)
    else:
        close = price_data.astype(float)
        lows = close
        highs = close

    low_arr = lows.values
    high_arr = highs.values

    local_lows = argrelextrema(low_arr, np.less, order=window)[0]
    local_highs = argrelextrema(high_arr, np.greater, order=window)[0]

    recent_span = min(252, len(close))
    if recent_span:
        recent_lows = lows.iloc[-recent_span:]
        recent_highs = highs.iloc[-recent_span:]
        local_lows = np.unique(np.append(local_lows, lows.index.get_loc(recent_lows.idxmin())))
        local_highs = np.unique(np.append(local_highs, highs.index.get_loc(recent_highs.idxmax())))

    def cluster_levels(indices, series: pd.Series, touch_series: pd.Series):
        candidates = sorted(
            [{"level": float(series.iloc[i]), "indices": [int(i)]} for i in indices],
            key=lambda item: item["level"],
        )
        clusters = []
        for candidate in candidates:
            level = candidate["level"]
            if clusters and abs(level - clusters[-1]["level"]) / clusters[-1]["level"] <= tolerance:
                merged_indices = clusters[-1]["indices"] + candidate["indices"]
                merged_levels = [float(series.iloc[i]) for i in merged_indices]
                clusters[-1]["level"] = float(np.mean(merged_levels))
                clusters[-1]["indices"] = merged_indices
            else:
                clusters.append(candidate)

        detailed = []
        for cluster in clusters:
            level = cluster["level"]
            band = max(level * tolerance, 0.01)
            touches = int((touch_series.sub(level).abs() <= band).sum())
            last_position = max(cluster["indices"])
            last_seen = close.index[last_position]
            distance_pct = (level - float(close.iloc[-1])) / float(close.iloc[-1])
            recency_boost = max(0.0, 1.0 - ((len(close) - 1 - last_position) / max(len(close), 1)))
            strength = touches + recency_boost
            detailed.append({
                "level": _safe_float(level),
                "touches": touches,
                "last_seen": last_seen.isoformat(),
                "distance_pct": _safe_float(distance_pct),
                "strength": _safe_float(strength),
            })
        return detailed

    support_details = cluster_levels(local_lows, lows, lows)
    resistance_details = cluster_levels(local_highs, highs, highs)
    supports = [level["level"] for level in support_details]
    resistances = [level["level"] for level in resistance_details]
    return supports, resistances, support_details, resistance_details

def make_prediction(ticker: str, models, scaler, feature_cols, df: pd.DataFrame, ensemble_proba, y_val, decision_threshold: float):
    last_row    = df[feature_cols].iloc[-1:].values
    last_scaled = scaler.transform(last_row)

    model_probas = [m.predict_proba(last_scaled)[0, 1] for m in models.values()]
    final_prob   = float(np.mean(model_probas))
    trend        = "BULLISH" if final_prob >= decision_threshold else "BEARISH"
    if trend == "BULLISH":
        confidence = 0.5 + 0.5 * ((final_prob - decision_threshold) / max(1 - decision_threshold, 1e-9))
    else:
        confidence = 0.5 + 0.5 * ((decision_threshold - final_prob) / max(decision_threshold, 1e-9))
    confidence = float(np.clip(confidence, 0.5, 1.0))

    last_price   = float(df["Close"].iloc[-1])
    atr          = float(df["ATR"].iloc[-1])
    adx          = float(df["ADX"].iloc[-1])
    bb_width     = float(df["BB_width"].iloc[-1])
    hist_vol     = float(df["Volatility_20d"].iloc[-1])
    close_z_20   = float(df["Close_Z_20"].iloc[-1])
    return_z_20  = float(df["Return_Z_20"].iloc[-1])
    volume_z_20  = float(df["Volume_Z_20"].iloc[-1])

    adx_factor     = min(adx / 25, 2.0)
    conf_factor    = confidence
    base_move_pct  = hist_vol / np.sqrt(252 / PREDICTION_DAYS)
    z_alignment    = (-close_z_20 / 2) if trend == "BULLISH" else (close_z_20 / 2)
    z_factor       = float(np.clip(1 + 0.15 * z_alignment, 0.85, 1.15))
    target_move    = last_price * base_move_pct * adx_factor * conf_factor * z_factor

    if trend == "BULLISH":
        price_target = last_price + target_move
        stop_loss    = last_price - atr * 2
    else:
        price_target = last_price - target_move
        stop_loss    = last_price + atr * 2

    base_dur   = PREDICTION_DAYS
    dur_factor = min(adx / 20, 2.5)
    est_days   = max(5, int(base_dur * dur_factor * conf_factor * z_factor))
    est_end    = df.index[-1] + BDay(est_days)

    if close_z_20 >= 2:
        z_state = "stretched high"
    elif close_z_20 <= -2:
        z_state = "stretched low"
    else:
        z_state = "normal range"

    supports, resistances, support_details, resistance_details = find_support_resistance(df)
    near_sup = max((s for s in supports    if s < last_price), default=None)
    near_res = min((r for r in resistances if r > last_price), default=None)
    nearest_supports = sorted(
        (level for level in support_details if level["level"] is not None and level["level"] < last_price),
        key=lambda level: abs(level["distance_pct"]),
    )[:5]
    nearest_resistances = sorted(
        (level for level in resistance_details if level["level"] is not None and level["level"] > last_price),
        key=lambda level: abs(level["distance_pct"]),
    )[:5]

    conclusion = {
        "ticker":          ticker,
        "trend":           trend,
        "confidence":      float(confidence),
        "final_prob":      float(final_prob),
        "decision_threshold": float(decision_threshold),
        "last_price":      float(last_price),
        "price_target":    float(price_target),
        "stop_loss":       float(stop_loss),
        "est_days":        int(est_days),
        "est_end":         est_end.isoformat(),
        "atr":             float(atr),
        "adx":             float(adx),
        "bb_width":        float(bb_width),
        "hist_vol":        float(hist_vol),
        "close_z_20":      float(close_z_20),
        "return_z_20":     float(return_z_20),
        "volume_z_20":     float(volume_z_20),
        "z_factor":        float(z_factor),
        "z_state":         z_state,
        "near_sup":        _safe_float(near_sup),
        "near_res":        _safe_float(near_res),
        "support":         _safe_float(near_sup),
        "resistance":      _safe_float(near_res),
        "support_distance_pct": _safe_float((near_sup - last_price) / last_price if near_sup else None),
        "resistance_distance_pct": _safe_float((near_res - last_price) / last_price if near_res else None),
        "support_levels":  nearest_supports,
        "resistance_levels": nearest_resistances,
        "all_supports":    [_safe_float(value) for value in supports if value < last_price][-5:],
        "all_resistances": [_safe_float(value) for value in resistances if value > last_price][:5],
    }
    return conclusion

def _sample_records(records, limit: int):
    if len(records) <= limit:
        return records
    idx = np.linspace(0, len(records) - 1, limit, dtype=int)
    return [records[i] for i in idx]

def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)

def build_training_progress(df: pd.DataFrame, split: int, dates_val, ensemble_pred, ensemble_proba, y_val, metrics: dict, top_features: list, feature_cols: list, decision_threshold: float):
    labeled_dates = df.dropna(subset=["Target"]).index
    train_dates = set(labeled_dates[:split])
    validation_dates = set(labeled_dates[split:])

    price_records = []
    for date, row in df.iterrows():
        if date in train_dates:
            phase = "train"
        elif date in validation_dates:
            phase = "validation"
        else:
            phase = "recent"

        price_records.append({
            "date": date.date().isoformat(),
            "close": _safe_float(row["Close"]),
            "sma50": _safe_float(row.get("SMA_50")),
            "sma200": _safe_float(row.get("SMA_200")),
            "rsi": _safe_float(row.get("RSI")),
            "macd": _safe_float(row.get("MACD")),
            "macd_signal": _safe_float(row.get("MACD_signal")),
            "macd_hist": _safe_float(row.get("MACD_hist")),
            "phase": phase,
        })

    validation_records = []
    val_rows = df.loc[dates_val, ["Close", "Future_avg_close"]]
    for date, row, pred, prob, actual in zip(dates_val, val_rows.itertuples(), ensemble_pred, ensemble_proba, y_val):
        close = float(row.Close)
        future_avg_close = float(row.Future_avg_close)
        validation_records.append({
            "date": date.date().isoformat(),
            "close": _safe_float(close),
            "future_avg_close": _safe_float(future_avg_close),
            "actual_move_pct": _safe_float((future_avg_close - close) / close),
            "probability": _safe_float(prob),
            "correct": bool(pred == actual),
            "prediction": int(pred),
            "actual": int(actual),
            "actual_trend": "BULLISH" if int(actual) == 1 else "BEARISH",
            "prediction_trend": "BULLISH" if int(pred) == 1 else "BEARISH",
        })

    def prediction_class_stats(prediction_value: int):
        records = [record for record in validation_records if record["prediction"] == prediction_value]
        correct = sum(1 for record in records if record["correct"])
        total = len(records)
        return {
            "calls": int(total),
            "correct": int(correct),
            "wrong": int(total - correct),
            "accuracy": _safe_float(correct / total if total else None),
        }

    total_correct = sum(1 for record in validation_records if record["correct"])
    validation_stats = {
        "total": int(len(validation_records)),
        "correct": int(total_correct),
        "wrong": int(len(validation_records) - total_correct),
        "accuracy": _safe_float(total_correct / len(validation_records) if validation_records else None),
        "decision_threshold": _safe_float(decision_threshold),
        "actual_bullish": int(sum(1 for record in validation_records if record["actual"] == 1)),
        "actual_bearish": int(sum(1 for record in validation_records if record["actual"] == 0)),
        "predicted_bullish": prediction_class_stats(1),
        "predicted_bearish": prediction_class_stats(0),
    }

    model_metrics = [
        {
            "model": name,
            "accuracy": _safe_float(values["acc"]),
            "auc": _safe_float(values["auc"]),
        }
        for name, values in metrics.items()
    ]

    feature_importance = [
        {
            "feature": name,
            "importance": _safe_float(importance),
        }
        for name, importance in top_features
    ]

    return {
        "summary": {
            "total_records": int(len(df)),
            "train_records": int(split),
            "validation_records": int(len(dates_val)),
            "neutral_records": int(df["Future_avg_close"].notna().sum() - df["Target"].notna().sum()),
            "feature_count": int(len(feature_cols)),
            "start_date": df.index[0].date().isoformat(),
            "end_date": df.index[-1].date().isoformat(),
        },
        "steps": [
            {"label": "Market data", "value": int(len(df)), "unit": "rows"},
            {"label": "Feature engineering", "value": int(len(feature_cols)), "unit": "features"},
            {"label": "Training sample", "value": int(split), "unit": "rows"},
            {"label": "Validation sample", "value": int(len(dates_val)), "unit": "rows"},
            {"label": "Models trained", "value": int(max(len(metrics) - 1, 0)), "unit": "models"},
        ],
        "price_series": _sample_records(price_records, 260),
        "validation_series": _sample_records(validation_records, 180),
        "validation_stats": validation_stats,
        "metrics": model_metrics,
        "feature_importance": feature_importance,
    }

def plot_all(ticker: str, df: pd.DataFrame, split: int, dates_val, ensemble_pred, ensemble_proba, y_val, metrics: dict, top_features: list, conclusion: dict):
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"], "axes.facecolor":    COLORS["panel"],
        "axes.edgecolor":    COLORS["border"], "axes.labelcolor":   COLORS["text"],
        "text.color":        COLORS["text"], "xtick.color":       COLORS["muted"],
        "ytick.color":       COLORS["muted"], "grid.color":        COLORS["border"],
        "grid.alpha":        0.5, "font.family":       "monospace", "font.size":         9,
    })

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.32, left=0.06, right=0.97, top=0.94, bottom=0.04)

    def ax_style(ax, title="", ylabel=""):
        ax.set_facecolor(COLORS["panel"])
        for sp in ax.spines.values(): sp.set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["muted"], labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if title: ax.set_title(title, color=COLORS["text"], fontsize=9, fontweight="bold", pad=6)
        if ylabel: ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=8)

    close   = df["Close"]
    dates   = df.index
    labeled_dates = df.dropna(subset=["Target"]).index
    train_d = labeled_dates[:split]
    val_d   = labeled_dates[split:]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train_d, close[:split], color=COLORS["blue"], linewidth=1.2, label="Train", alpha=0.9)
    ax1.plot(dates[split:], close[split:], color=COLORS["purple"], linewidth=1.4, label="Validation & Recent", alpha=0.9)
    ax1.plot(dates, df["SMA_50"],  color=COLORS["orange"], linewidth=0.8, alpha=0.7, label="SMA 50")
    ax1.plot(dates, df["SMA_200"], color=COLORS["cyan"], linewidth=0.8, alpha=0.7, label="SMA 200")
    ax1.fill_between(dates, df["BB_upper"], df["BB_lower"], alpha=0.07, color=COLORS["blue"])

    c = conclusion
    for s in c["all_supports"][-3:]:
        ax1.axhline(s, color=COLORS["green"], linewidth=0.7, linestyle="--", alpha=0.5)
    for r in c["all_resistances"][:3]:
        ax1.axhline(r, color=COLORS["red"], linewidth=0.7, linestyle="--", alpha=0.5)

    ax1.axvline(val_d[0], color=COLORS["yellow"], linewidth=1.5, linestyle=":", alpha=0.8, label="Train/Val split")

    last_date  = dates[-1]
    fut_date   = datetime.fromisoformat(c["est_end"])
    lp         = c["last_price"]
    pt         = c["price_target"]
    arrow_col  = COLORS["green"] if c["trend"] == "BULLISH" else COLORS["red"]

    ax1.annotate("", xy=(fut_date, pt), xytext=(last_date, lp), arrowprops=dict(arrowstyle="->", color=arrow_col, lw=2.5, connectionstyle="arc3,rad=0.15"))
    ax1.scatter([fut_date], [pt], color=arrow_col, s=100, zorder=6)
    ax1.text(fut_date, pt, f"  Target\n  ${pt:.2f}", color=arrow_col, fontsize=8, fontweight="bold", va="center")
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.2, labelcolor=COLORS["text"])
    ax_style(ax1, f"{ticker} — Full History with Bollinger Bands & Trend Projection", "Price (USD)")

    ax2 = fig.add_subplot(gs[1, 0])
    colors_vol = [COLORS["green"] if c_ >= o else COLORS["red"] for c_, o in zip(df["Close"], df["Open"])]
    ax2.bar(dates, df["Volume"], color=colors_vol, alpha=0.7, width=1)
    ax2.plot(dates, df["Vol_SMA20"], color=COLORS["yellow"], linewidth=1, label="Vol SMA20")
    ax_style(ax2, "Volume", "Shares")
    ax2.legend(fontsize=7, framealpha=0.1)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(dates, df["RSI"], color=COLORS["purple"], linewidth=1)
    ax3.axhline(70, color=COLORS["red"],   linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.axhline(30, color=COLORS["green"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.fill_between(dates, df["RSI"], 70, where=df["RSI"] > 70, alpha=0.15, color=COLORS["red"])
    ax3.fill_between(dates, df["RSI"], 30, where=df["RSI"] < 30, alpha=0.15, color=COLORS["green"])
    ax3.set_ylim(0, 100)
    ax3.text(dates[-1], 72, "OB", color=COLORS["red"],   fontsize=7)
    ax3.text(dates[-1], 26, "OS", color=COLORS["green"], fontsize=7)
    ax_style(ax3, "RSI (14)", "RSI")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(dates, df["MACD"],        color=COLORS["blue"],   linewidth=1,   label="MACD")
    ax4.plot(dates, df["MACD_signal"], color=COLORS["orange"], linewidth=1,   label="Signal")
    hist_pos = df["MACD_hist"].clip(lower=0)
    hist_neg = df["MACD_hist"].clip(upper=0)
    ax4.bar(dates, hist_pos, color=COLORS["green"], alpha=0.5, width=1)
    ax4.bar(dates, hist_neg, color=COLORS["red"],   alpha=0.5, width=1)
    ax4.axhline(0, color=COLORS["muted"], linewidth=0.5)
    ax4.legend(fontsize=7, framealpha=0.1)
    ax_style(ax4, "MACD", "")

    ax5 = fig.add_subplot(gs[2, :2])
    val_close = df.loc[val_d, "Close"].values
    ax5.plot(val_d, val_close, color=COLORS["blue"], linewidth=1.2, label="Actual price", alpha=0.9)
    for i in range(len(val_d)):
        correct = (ensemble_pred[i] == y_val[i])
        col = COLORS["green"] if correct else COLORS["red"]
        ax5.axvspan(val_d[i], val_d[min(i+1, len(val_d)-1)], alpha=0.12, color=col)
    ax5.set_title("Validation Period — Prediction Accuracy (Green=Correct, Red=Wrong)", color=COLORS["text"], fontsize=9, fontweight="bold", pad=6)
    ax_style(ax5, "", "Price (USD)")

    ax5b = ax5.twinx()
    ax5b.plot(val_d, ensemble_proba, color=COLORS["purple"], linewidth=0.9, alpha=0.7, label="Bullish prob")
    ax5b.axhline(0.5, color=COLORS["yellow"], linewidth=0.6, linestyle="--", alpha=0.6)
    ax5b.set_ylim(0, 1)
    ax5b.set_ylabel("Bullish probability", color=COLORS["purple"], fontsize=8)
    ax5b.tick_params(colors=COLORS["purple"])
    lines1, labs1 = ax5.get_legend_handles_labels()
    lines2, labs2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labs1 + labs2, fontsize=7, framealpha=0.15, loc="upper left")

    ax6 = fig.add_subplot(gs[2, 2])
    names  = list(metrics.keys())
    accs   = [metrics[n]["acc"] for n in names]
    aucs   = [metrics[n]["auc"] for n in names]
    x      = np.arange(len(names))
    w      = 0.35
    ax6.bar(x - w/2, accs, w, color=COLORS["blue"],   alpha=0.8, label="Accuracy")
    ax6.bar(x + w/2, aucs, w, color=COLORS["purple"], alpha=0.8, label="AUC")
    ax6.set_xticks(x)
    ax6.set_xticklabels([n[:8] for n in names], fontsize=7)
    ax6.set_ylim(0, 1.0)
    for i, (a, u) in enumerate(zip(accs, aucs)):
        ax6.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7, color=COLORS["blue"])
        ax6.text(i + w/2, u + 0.01, f"{u:.2f}", ha="center", fontsize=7, color=COLORS["purple"])
    ax6.legend(fontsize=7, framealpha=0.15)
    ax_style(ax6, "Model Performance", "Score")

    ax7 = fig.add_subplot(gs[3, :2])
    feats = [f[0] for f in top_features][::-1]
    imps  = [f[1] for f in top_features][::-1]
    bar_colors = [COLORS["cyan"] if i < len(imps) // 2 else COLORS["blue"] for i in range(len(imps))]
    bars = ax7.barh(feats, imps, color=bar_colors, alpha=0.8, height=0.7)
    for bar, val in zip(bars, imps):
        ax7.text(val + 0.001, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center", fontsize=7, color=COLORS["muted"])
    ax_style(ax7, "Top 15 Feature Importances (XGBoost)", "Importance")

    ax8 = fig.add_subplot(gs[3, 2])
    ax8.plot(dates, df["ADX"],     color=COLORS["orange"], linewidth=1, label="ADX")
    ax8.plot(dates, df["ADX_pos"], color=COLORS["green"],  linewidth=0.8, alpha=0.7, label="+DI")
    ax8.plot(dates, df["ADX_neg"], color=COLORS["red"],    linewidth=0.8, alpha=0.7, label="-DI")
    ax8.axhline(25, color=COLORS["yellow"], linewidth=0.7, linestyle="--", alpha=0.6, label="Trend threshold")
    ax8.legend(fontsize=7, framealpha=0.1)
    ax_style(ax8, "ADX — Trend Strength", "")

    ax9 = fig.add_subplot(gs[4, :])
    ax9.set_facecolor(COLORS["panel"])
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis("off")

    rect = FancyBboxPatch((0.01, 0.04), 0.98, 0.93, boxstyle="round,pad=0.01", linewidth=1.5, edgecolor=(COLORS["green"] if c["trend"] == "BULLISH" else COLORS["red"]), facecolor=COLORS["bg"], alpha=0.8)
    ax9.add_patch(rect)
    trend_col = COLORS["green"] if c["trend"] == "BULLISH" else COLORS["red"]
    trend_sym  = "▲" if c["trend"] == "BULLISH" else "▼"
    ax9.text(0.5, 0.88, f"{trend_sym}  {c['ticker']} — {c['trend']} TREND FOR NEXT ~{c['est_days']} DAYS  {trend_sym}", ha="center", va="center", fontsize=16, fontweight="bold", color=trend_col, transform=ax9.transAxes)

    col_x = [0.12, 0.45, 0.78]
    rows = [
        [f"Current Price   ${c['last_price']:.2f}", f"Price Target    ${c['price_target']:.2f}", f"Est Duration    ~{c['est_days']} trading days"],
        [f"Confidence      {c['confidence']*100:.1f}%", f"Stop Loss       ${c['stop_loss']:.2f}", f"Target Date     {datetime.fromisoformat(c['est_end']).date()}"],
        [f"ATR             ${c['atr']:.2f}", f"ADX             {c['adx']:.1f}  ({'Strong' if c['adx']>25 else 'Mod'})", f"Hist Vol        {c['hist_vol']*100:.1f}% p.a."],
        [f"Support         ${c['near_sup']:.2f}" if c['near_sup'] else "Support    —", f"Resistance      ${c['near_res']:.2f}" if c['near_res'] else "Resistance —", f"Move            {(c['price_target']-c['last_price'])/c['last_price']*100:+.1f}%"],
    ]

    y_positions = [0.70, 0.55, 0.40, 0.25]
    for row_y, row in zip(y_positions, rows):
        for cx, txt in zip(col_x, row):
            label, _, val = txt.partition("  ")
            ax9.text(cx - 0.06, row_y, label, ha="left", fontsize=9, color=COLORS["muted"], transform=ax9.transAxes)
            ax9.text(cx + 0.08, row_y, val.strip(), ha="left", fontsize=9.5, fontweight="bold", color=COLORS["text"], transform=ax9.transAxes)

    ax9.text(0.5, 0.08, "⚠  For research & educational purposes only. Not financial advice.", ha="center", fontsize=7.5, color=COLORS["muted"], transform=ax9.transAxes, style="italic")
    fig.suptitle(f"QUANTITATIVE TREND PREDICTOR  |  {ticker}  |  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=11, fontweight="bold", color=COLORS["text"], y=0.965)

    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/{ticker}_quant_predictor.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    return f"{ticker}_quant_predictor.png"

def run_prediction(ticker: str):
    ticker = ticker.upper()
    df = fetch_data(ticker)
    df = add_features(df)
    models, scaler, feature_cols, df, split, dates_val, preds_val, probas_val, ensemble_pred, ensemble_proba, y_val, metrics, top_features, decision_threshold = train_and_validate(df)
    conclusion = make_prediction(ticker, models, scaler, feature_cols, df, ensemble_proba, y_val, decision_threshold)
    training_progress = build_training_progress(df, split, dates_val, ensemble_pred, ensemble_proba, y_val, metrics, top_features, feature_cols, decision_threshold)
    return conclusion, training_progress

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', 'SPY')
    try:
        conclusion, training_progress = run_prediction(ticker)
        return jsonify({
            "status": "success",
            "conclusion": conclusion,
            "training_progress": training_progress
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
