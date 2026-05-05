"""
========================================================
  QUANTITATIVE STOCK TREND PREDICTOR  v2.0
  - Fetches full historical data
  - Computes technical indicators + Z-Score suite
  - Hurst Exponent regime detection
  - Trains XGBoost + ensemble model
  - Validates on holdout set
  - Predicts trend direction, duration & price targets
  - Z-score adjusts confidence & duration dynamically
========================================================
"""
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from scipy.signal import argrelextrema
import textwrap

# ─────────────────────────────────────────────
# CONFIG  — change TICKER here
# ─────────────────────────────────────────────
TICKER          = "SMH"
PREDICTION_DAYS = 5           # forward window for weekly trend prediction
TRAIN_RATIO     = 0.80        # 80 % train / 20 % validation
LOOKBACK        = 20          # rolling-window base for features
SUPPORT_RES_WIN = 10          # local extrema window

# Z-score thresholds
Z_EXTREME       = 2.0         # |Z| > 2.0 → overextended, compress duration
Z_MODERATE      = 1.0         # |Z| < 1.0 → trend has room, extend duration
Z_WINDOWS       = [20, 50]    # rolling windows for price Z-score

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


# ══════════════════════════════════════════════
# 1. DATA FETCHING
# ══════════════════════════════════════════════

def fetch_data(ticker: str) -> pd.DataFrame:
    print(f"\n{'═'*60}")
    print(f"  QUANTITATIVE STOCK TREND PREDICTOR  v2.0")
    print(f"  Ticker : {ticker}")
    print(f"{'═'*60}")
    print("\n[1/5] Fetching historical data …")

    live_success = False
    df = pd.DataFrame()

    try:
        t = yf.Ticker(ticker)
        df = t.history(
            period="max",
            auto_adjust=True,
            repair=True,
            keepna=False,
        )

        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)

            last_price = df["Close"].iloc[-1]
            last_date  = df.index[-1].date()
            live_success = True

            print(f"  ✓  Live data fetched via Yahoo Finance")
            print(f"  ✓  {len(df):,} daily bars  |  "
                  f"{df.index[0].date()} → {last_date}")
            print(f"  ✓  Last close : ${last_price:.2f}  ({last_date})")
        else:
            print(f"  ⚠  Yahoo Finance returned empty data — falling back to simulation")

    except Exception as e:
        print(f"  ⚠  Yahoo Finance unavailable ({e}) — falling back to simulation")

    if not live_success:
        print(f"  ℹ  Generating realistic simulated data for '{ticker}' …")

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

        print(f"  ✓  {len(df):,} simulated bars  |  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        print(f"  ⚠  NOTE: Results based on simulation, NOT real market data")

    return df


# ══════════════════════════════════════════════
# 2. HURST EXPONENT  (regime detection)
# ══════════════════════════════════════════════

def hurst_exponent(series: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Hurst Exponent via R/S analysis.
    H < 0.5  → mean-reverting (anti-persistent)
    H ≈ 0.5  → random walk
    H > 0.5  → trending (persistent)
    """
    lags = range(min_lag, max_lag)
    try:
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        tau = np.array(tau)
        tau[tau == 0] = 1e-10
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    except Exception:
        return 0.5   # default to random walk if computation fails


def rolling_hurst(close: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Hurst exponent over a sliding window."""
    result = pd.Series(index=close.index, dtype=float)
    arr = close.values
    for i in range(window, len(arr)):
        result.iloc[i] = hurst_exponent(arr[i - window: i])
    result.iloc[:window] = np.nan
    return result


# ══════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (now includes Z-score)
# ══════════════════════════════════════════════

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Computing technical indicators, Z-scores & regime features …")

    # ── Trend / Moving Averages ──
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"]  = ta.trend.sma_indicator(df["Close"], window=w)
        df[f"EMA_{w}"]  = ta.trend.ema_indicator(df["Close"], window=w)

    # ── Momentum ──
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
    df["ROC_5"]         = ta.momentum.roc(df["Close"], window=5)

    # ── Volatility ──
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_upper"]      = bb.bollinger_hband()
    df["BB_mid"]        = bb.bollinger_mavg()
    df["BB_lower"]      = bb.bollinger_lband()
    df["BB_width"]      = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pct"]        = bb.bollinger_pband()
    df["ATR"]           = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    df["ATR_pct"]       = df["ATR"] / df["Close"]

    # ── Volume ──
    df["OBV"]           = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["Vol_SMA20"]     = df["Volume"].rolling(20).mean()
    df["Vol_ratio"]     = df["Volume"] / df["Vol_SMA20"]
    df["VWAP"]          = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["CMF"]           = ta.volume.chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"])

    # ── Price-derived ──
    df["Return_1d"]     = df["Close"].pct_change(1)
    df["Return_5d"]     = df["Close"].pct_change(5)
    df["Return_20d"]    = df["Close"].pct_change(20)
    df["Log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["High_Low_pct"]  = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_pct"]= (df["Close"] - df["Open"]) / df["Open"]

    # ── Trend strength ──
    adx_ind             = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"]           = adx_ind.adx()
    df["ADX_pos"]       = adx_ind.adx_pos()
    df["ADX_neg"]       = adx_ind.adx_neg()
    df["DI_diff"]       = df["ADX_pos"] - df["ADX_neg"]   # directional bias

    # ── Ichimoku Cloud ──
    ich = ta.trend.IchimokuIndicator(df["High"], df["Low"])
    df["Ichi_tenkan"]   = ich.ichimoku_conversion_line()
    df["Ichi_kijun"]    = ich.ichimoku_base_line()
    df["Ichi_span_a"]   = ich.ichimoku_a()
    df["Ichi_span_b"]   = ich.ichimoku_b()
    df["Ichi_above_cloud"] = (
        (df["Close"] > df["Ichi_span_a"]) &
        (df["Close"] > df["Ichi_span_b"])
    ).astype(int)
    df["Ichi_tk_cross"] = (df["Ichi_tenkan"] > df["Ichi_kijun"]).astype(int)

    # ── Rolling stats ──
    for w in [5, 10, 20]:
        df[f"Volatility_{w}d"] = df["Log_return"].rolling(w).std() * np.sqrt(252)
        df[f"High_{w}d"]       = df["High"].rolling(w).max()
        df[f"Low_{w}d"]        = df["Low"].rolling(w).min()

    # ══════════════════════════════════════════
    # Z-SCORE SUITE
    # ══════════════════════════════════════════

    # 1. Price Z-score vs rolling mean (identifies overextension)
    for w in Z_WINDOWS:
        roll_mean = df["Close"].rolling(w).mean()
        roll_std  = df["Close"].rolling(w).std()
        df[f"Z_price_{w}"]  = (df["Close"] - roll_mean) / roll_std.replace(0, np.nan)

    # 2. RSI Z-score — is RSI itself overextended vs its own history?
    df["Z_RSI"]             = (df["RSI"] - df["RSI"].rolling(50).mean()) / \
                               df["RSI"].rolling(50).std().replace(0, np.nan)

    # 3. Volume Z-score — abnormal volume spikes / droughts
    df["Z_volume"]          = (df["Volume"] - df["Volume"].rolling(20).mean()) / \
                               df["Volume"].rolling(20).std().replace(0, np.nan)

    # 4. ATR Z-score — volatility regime (rising vs falling vol)
    df["Z_ATR"]             = (df["ATR"] - df["ATR"].rolling(50).mean()) / \
                               df["ATR"].rolling(50).std().replace(0, np.nan)

    # 5. MACD Histogram Z-score — momentum acceleration extremes
    df["Z_MACD_hist"]       = (df["MACD_hist"] - df["MACD_hist"].rolling(50).mean()) / \
                               df["MACD_hist"].rolling(50).std().replace(0, np.nan)

    # 6. ROC Z-score — rate-of-change anomaly
    df["Z_ROC"]             = (df["ROC"] - df["ROC"].rolling(50).mean()) / \
                               df["ROC"].rolling(50).std().replace(0, np.nan)

    # 7. BB_pct Z-score — how extreme is price within its Bollinger Band?
    df["Z_BB_pct"]          = (df["BB_pct"] - df["BB_pct"].rolling(50).mean()) / \
                               df["BB_pct"].rolling(50).std().replace(0, np.nan)

    # 8. COMPOSITE Z-score — weighted blend across all Z signals
    #    Weights reflect predictive priority for trend continuation
    z_components = {
        f"Z_price_{Z_WINDOWS[0]}": 0.30,   # primary overextension signal
        f"Z_price_{Z_WINDOWS[1]}": 0.20,   # longer-term context
        "Z_MACD_hist":             0.15,   # momentum
        "Z_RSI":                   0.15,   # momentum
        "Z_volume":                0.10,   # volume confirmation
        "Z_ATR":                   0.10,   # volatility regime
    }
    df["Z_composite"] = sum(
        df[col] * w for col, w in z_components.items()
    )
    # Clip to prevent outlier blow-up
    df["Z_composite"] = df["Z_composite"].clip(-4, 4)

    # 9. Z-score regime labels (features for the model)
    df["Z_overextended"]    = (df["Z_composite"].abs() > Z_EXTREME).astype(int)
    df["Z_trending"]        = (df["Z_composite"].abs() < Z_MODERATE).astype(int)
    df["Z_direction"]       = np.sign(df["Z_composite"])  # +1 extended up, -1 extended down

    # ══════════════════════════════════════════
    # HURST EXPONENT  (rolling regime)
    # ══════════════════════════════════════════
    print("  ℹ  Computing rolling Hurst exponent (takes a moment) …")
    df["Hurst"] = rolling_hurst(df["Close"], window=100)
    df["Hurst_trending"]     = (df["Hurst"] > 0.55).astype(int)
    df["Hurst_mean_rev"]     = (df["Hurst"] < 0.45).astype(int)

    # ── TARGET: will price be higher in PREDICTION_DAYS? ──
    df["Future_close"]      = df["Close"].shift(-PREDICTION_DAYS)
    df["Target"]            = (df["Future_close"] > df["Close"]).astype(int)

    feature_cols = [c for c in df.columns if c not in ("Future_close", "Target")]
    df.dropna(subset=feature_cols, inplace=True)
    print(f"  ✓  {df.shape[1]} columns  |  {len(df):,} usable rows after dropna")

    # Report Z-score summary
    z20 = df["Z_price_20"].iloc[-1]
    z50 = df["Z_price_50"].iloc[-1]
    zc  = df["Z_composite"].iloc[-1]
    h   = df["Hurst"].iloc[-1]
    print(f"\n  Z-SCORE SNAPSHOT (latest bar):")
    print(f"    Z_price_20    = {z20:+.2f}  {'⚠ OVEREXTENDED' if abs(z20) > Z_EXTREME else '✓ normal'}")
    print(f"    Z_price_50    = {z50:+.2f}  {'⚠ OVEREXTENDED' if abs(z50) > Z_EXTREME else '✓ normal'}")
    print(f"    Z_composite   = {zc:+.2f}  ({'overextended' if abs(zc) > Z_EXTREME else 'moderate' if abs(zc) > Z_MODERATE else 'neutral'})")
    print(f"    Hurst (100d)  = {h:.3f}   ({'TRENDING' if h > 0.55 else 'MEAN-REV' if h < 0.45 else 'RANDOM WALK'})")

    return df


# ══════════════════════════════════════════════
# 4. MODEL TRAINING & VALIDATION
# ══════════════════════════════════════════════
def train_and_validate(df: pd.DataFrame):
    print("\n[3/5] Training & validating model …")

    train_df = df.dropna(subset=["Future_close"])

    feature_cols = [c for c in df.columns
                    if c not in ("Open","High","Low","Close","Volume",
                                 "Future_close","Target")]
    X = train_df[feature_cols].values
    y = train_df["Target"].values
    dates = train_df.index

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    dates_val       = dates[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # ── Ensemble ──
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42),
    }

    preds_val  = {}
    probas_val = {}
    metrics    = {}

    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        p = mdl.predict(X_val_s)
        pr = mdl.predict_proba(X_val_s)[:, 1]
        preds_val[name]  = p
        probas_val[name] = pr
        acc = accuracy_score(y_val, p)
        auc = roc_auc_score(y_val, pr)
        metrics[name] = {"acc": acc, "auc": auc}
        print(f"  {name:<14}  Acc={acc:.3f}  AUC={auc:.3f}")

    # ── Ensemble average ──
    ensemble_proba = np.mean([probas_val[n] for n in models], axis=0)
    ensemble_pred  = (ensemble_proba >= 0.5).astype(int)
    ens_acc = accuracy_score(y_val, ensemble_pred)
    ens_auc = roc_auc_score(y_val, ensemble_proba)
    metrics["Ensemble"] = {"acc": ens_acc, "auc": ens_auc}
    print(f"  {'Ensemble':<14}  Acc={ens_acc:.3f}  AUC={ens_auc:.3f}  ← best")

    # Refit on full data for prediction
    for name, mdl in models.items():
        mdl.fit(scaler.transform(X), y)

    # Feature importances from XGBoost
    fi = dict(zip(feature_cols, models["XGBoost"].feature_importances_))
    top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]

    return (models, scaler, feature_cols,
            df, split, dates_val,
            preds_val, probas_val, ensemble_pred, ensemble_proba,
            y_val, metrics, top_features)


# ══════════════════════════════════════════════
# 5. SUPPORT / RESISTANCE
# ══════════════════════════════════════════════
def find_support_resistance(close: pd.Series, window: int = SUPPORT_RES_WIN):
    arr = close.values
    lmin = argrelextrema(arr, np.less,    order=window)[0]
    lmax = argrelextrema(arr, np.greater, order=window)[0]

    def cluster(indices, tol=0.015):
        levels = sorted(arr[i] for i in indices)
        clustered = []
        for lv in levels:
            if not clustered or abs(lv - clustered[-1]) / clustered[-1] > tol:
                clustered.append(lv)
        return clustered

    return cluster(lmin), cluster(lmax)


# ══════════════════════════════════════════════
# 6. PREDICTION & Z-SCORE ADJUSTED CONCLUSION
# ══════════════════════════════════════════════
def make_prediction(models, scaler, feature_cols,
                    df: pd.DataFrame, ensemble_proba, y_val):

    last_row    = df[feature_cols].iloc[-1:].values
    last_scaled = scaler.transform(last_row)

    model_probas = [m.predict_proba(last_scaled)[0, 1] for m in models.values()]
    final_prob   = float(np.mean(model_probas))
    trend        = "BULLISH" if final_prob >= 0.5 else "BEARISH"
    raw_conf     = final_prob if trend == "BULLISH" else 1 - final_prob

    # ── Current market state ──
    last_price   = float(df["Close"].iloc[-1])
    atr          = float(df["ATR"].iloc[-1])
    adx          = float(df["ADX"].iloc[-1])
    bb_width     = float(df["BB_width"].iloc[-1])
    hist_vol     = float(df["Volatility_20d"].iloc[-1])
    hurst        = float(df["Hurst"].iloc[-1]) if not np.isnan(df["Hurst"].iloc[-1]) else 0.5

    # ── Z-score values ──
    z_price_20   = float(df["Z_price_20"].iloc[-1])
    z_price_50   = float(df["Z_price_50"].iloc[-1])
    z_composite  = float(df["Z_composite"].iloc[-1])
    z_volume     = float(df["Z_volume"].iloc[-1])
    z_atr        = float(df["Z_ATR"].iloc[-1])
    z_rsi        = float(df["Z_RSI"].iloc[-1])
    z_macd       = float(df["Z_MACD_hist"].iloc[-1])

    print(f"\n  DEBUG — Last price : ${last_price:.2f}  |  Date : {df.index[-1].date()}")

    # ══════════════════════════════════════════
    # Z-SCORE ADJUSTMENTS TO CONFIDENCE & DURATION
    # ══════════════════════════════════════════

    # --- Confidence adjustment ---
    # When Z-composite opposes the trend direction, trim confidence.
    # When Z-composite aligns with trend direction, boost confidence.
    z_sign       = np.sign(z_composite) if z_composite != 0 else 0
    trend_sign   = 1 if trend == "BULLISH" else -1

    if abs(z_composite) > Z_EXTREME:
        # Overextended — price likely to mean-revert
        if z_sign == trend_sign:
            # e.g. BULLISH but Z is very high → overextended upward → reduce confidence
            conf_z_adj = -0.08 * min(abs(z_composite) / Z_EXTREME, 2.0)
        else:
            # e.g. BULLISH but Z is very negative → deep oversold → boost confidence
            conf_z_adj = +0.05 * min(abs(z_composite) / Z_EXTREME, 2.0)
    elif abs(z_composite) < Z_MODERATE:
        # Neutral zone — trend has room to breathe
        conf_z_adj = +0.03
    else:
        conf_z_adj = 0.0

    # Hurst regime boost/penalty
    if hurst > 0.55 and z_sign == trend_sign:
        hurst_adj = +0.04   # trending regime, aligned → boost
    elif hurst < 0.45:
        hurst_adj = -0.04   # mean-reverting regime → penalize trend continuation
    else:
        hurst_adj = 0.0

    confidence = float(np.clip(raw_conf + conf_z_adj + hurst_adj, 0.51, 0.95))

    # --- Duration adjustment ---
    # Base: ADX-driven (as before)
    adx_factor  = min(adx / 25, 2.0)
    base_dur    = PREDICTION_DAYS
    base_days   = max(5, int(base_dur * adx_factor * raw_conf))

    # Z-score duration modifier
    if abs(z_composite) > Z_EXTREME:
        # Overextended → expect shorter trend before reversal
        z_dur_factor = max(0.40, 1.0 - 0.20 * (abs(z_composite) - Z_EXTREME))
    elif abs(z_composite) < Z_MODERATE:
        # Plenty of room → expect longer trend
        z_dur_factor = min(1.40, 1.0 + 0.20 * (Z_MODERATE - abs(z_composite)))
    else:
        z_dur_factor = 1.0

    # Hurst regime duration modifier
    if hurst > 0.55:
        hurst_dur_factor = 1.20   # trending market → runs longer
    elif hurst < 0.45:
        hurst_dur_factor = 0.70   # mean-reverting → shorter trend
    else:
        hurst_dur_factor = 1.00

    est_days = max(5, int(base_days * z_dur_factor * hurst_dur_factor))
    est_end  = df.index[-1] + timedelta(days=est_days)

    # ── Target price ──
    base_move_pct = hist_vol / np.sqrt(252 / est_days)
    target_move   = last_price * base_move_pct * adx_factor * confidence

    if trend == "BULLISH":
        price_target = last_price + target_move
        stop_loss    = last_price - atr * 2
    else:
        price_target = last_price - target_move
        stop_loss    = last_price + atr * 2

    # ── Support / Resistance ──
    supports, resistances = find_support_resistance(df["Close"])
    near_sup = max((s for s in supports    if s < last_price), default=None)
    near_res = min((r for r in resistances if r > last_price), default=None)

    # Regime label
    if hurst > 0.55:
        regime = "TRENDING"
    elif hurst < 0.45:
        regime = "MEAN-REVERTING"
    else:
        regime = "RANDOM WALK"

    conclusion = {
        "ticker":          TICKER,
        "trend":           trend,
        "confidence":      confidence,
        "raw_confidence":  raw_conf,
        "final_prob":      final_prob,
        "conf_z_adj":      conf_z_adj,
        "hurst_adj":       hurst_adj,
        "last_price":      last_price,
        "price_target":    price_target,
        "stop_loss":       stop_loss,
        "est_days":        est_days,
        "est_end":         est_end,
        "base_days":       base_days,
        "z_dur_factor":    z_dur_factor,
        "hurst_dur_factor":hurst_dur_factor,
        "atr":             atr,
        "adx":             adx,
        "bb_width":        bb_width,
        "hist_vol":        hist_vol,
        "hurst":           hurst,
        "regime":          regime,
        "z_price_20":      z_price_20,
        "z_price_50":      z_price_50,
        "z_composite":     z_composite,
        "z_volume":        z_volume,
        "z_atr":           z_atr,
        "z_rsi":           z_rsi,
        "z_macd":          z_macd,
        "near_sup":        near_sup,
        "near_res":        near_res,
        "all_supports":    supports[-5:],
        "all_resistances": resistances[:5],
    }
    return conclusion


# ══════════════════════════════════════════════
# 7. CONSOLE PRINT
# ══════════════════════════════════════════════
def print_conclusion(c: dict, metrics: dict):
    print(f"\n{'═'*60}")
    print(f"  PREDICTION SUMMARY  —  {c['ticker']}")
    print(f"{'═'*60}")
    trend_sym = "▲ BULLISH" if c["trend"] == "BULLISH" else "▼ BEARISH"
    print(f"  Trend        : {trend_sym}")
    print(f"  Confidence   : {c['confidence']*100:.1f}%  "
          f"(raw={c['raw_confidence']*100:.1f}%, "
          f"Z-adj={c['conf_z_adj']:+.3f}, "
          f"Hurst-adj={c['hurst_adj']:+.3f})")
    print(f"  Last Price   : ${c['last_price']:.2f}")
    print(f"  Price Target : ${c['price_target']:.2f}  "
          f"({'↑' if c['trend']=='BULLISH' else '↓'}"
          f" {abs(c['price_target']-c['last_price'])/c['last_price']*100:.1f}%)")
    print(f"  Stop Loss    : ${c['stop_loss']:.2f}")
    print(f"  Est Duration : ~{c['est_days']} trading days  "
          f"(base={c['base_days']}d, "
          f"Z-factor={c['z_dur_factor']:.2f}, "
          f"Hurst-factor={c['hurst_dur_factor']:.2f})")
    print(f"  Target Date  : ≈ {c['est_end'].date()}")
    print(f"  ADX Strength : {c['adx']:.1f}  "
          f"({'Strong' if c['adx']>25 else 'Moderate' if c['adx']>20 else 'Weak'} trend)")
    print(f"  ATR (vol)    : ${c['atr']:.2f}")
    print(f"  Hist Vol     : {c['hist_vol']*100:.1f}% annualised")
    print(f"  Hurst (100d) : {c['hurst']:.3f}  → {c['regime']}")

    print(f"\n  Z-SCORE ANALYSIS:")
    def z_label(z):
        if abs(z) > Z_EXTREME:   return f"⚠ OVEREXTENDED ({'UP' if z>0 else 'DOWN'})"
        elif abs(z) > Z_MODERATE: return f"elevated ({'UP' if z>0 else 'DOWN'})"
        else:                     return "neutral"
    print(f"    Z_price_20  = {c['z_price_20']:+.2f}  {z_label(c['z_price_20'])}")
    print(f"    Z_price_50  = {c['z_price_50']:+.2f}  {z_label(c['z_price_50'])}")
    print(f"    Z_RSI       = {c['z_rsi']:+.2f}  {z_label(c['z_rsi'])}")
    print(f"    Z_volume    = {c['z_volume']:+.2f}  {z_label(c['z_volume'])}")
    print(f"    Z_ATR       = {c['z_atr']:+.2f}  {z_label(c['z_atr'])}")
    print(f"    Z_MACD_hist = {c['z_macd']:+.2f}  {z_label(c['z_macd'])}")
    print(f"    Z_COMPOSITE = {c['z_composite']:+.2f}  {z_label(c['z_composite'])}")

    if c["near_sup"]: print(f"\n  Support      : ${c['near_sup']:.2f}")
    if c["near_res"]: print(f"  Resistance   : ${c['near_res']:.2f}")
    print(f"\n  Model Performance (validation set):")
    for name, m in metrics.items():
        print(f"    {name:<14} Acc={m['acc']*100:.1f}%  AUC={m['auc']:.3f}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════
# 8. CHARTS
# ══════════════════════════════════════════════
def plot_all(df: pd.DataFrame, split: int, dates_val,
             ensemble_pred, ensemble_proba, y_val,
             metrics: dict, top_features: list,
             conclusion: dict):

    print("[5/5] Generating charts …")

    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["panel"],
        "axes.edgecolor":    COLORS["border"],
        "axes.labelcolor":   COLORS["text"],
        "text.color":        COLORS["text"],
        "xtick.color":       COLORS["muted"],
        "ytick.color":       COLORS["muted"],
        "grid.color":        COLORS["border"],
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "font.size":         9,
    })

    fig = plt.figure(figsize=(24, 34))
    fig.patch.set_facecolor(COLORS["bg"])

    # 6 rows: price, indicators row, validation row, z-score row, features+hurst, summary
    gs = gridspec.GridSpec(6, 3,
                           figure=fig,
                           hspace=0.48, wspace=0.32,
                           left=0.06, right=0.97,
                           top=0.95, bottom=0.03)

    def ax_style(ax, title="", ylabel=""):
        ax.set_facecolor(COLORS["panel"])
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["muted"], labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if title:
            ax.set_title(title, color=COLORS["text"],
                         fontsize=9, fontweight="bold", pad=6)
        if ylabel:
            ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=8)

    close   = df["Close"]
    dates   = df.index
    c       = conclusion

    labeled_dates = df.dropna(subset=["Future_close"]).index
    train_d = labeled_dates[:split]
    val_d   = labeled_dates[split:]

    # ─────────────────────────────────────────
    # CHART 1 — Full price + MAs (row 0, spans 3 cols)
    # ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train_d, close[:split], color=COLORS["blue"],
             linewidth=1.2, label="Train", alpha=0.9)
    ax1.plot(dates[split:], close[split:], color=COLORS["purple"],
             linewidth=1.4, label="Validation & Recent", alpha=0.9)
    ax1.plot(dates, df["SMA_50"],  color=COLORS["orange"],
             linewidth=0.8, alpha=0.7, label="SMA 50")
    ax1.plot(dates, df["SMA_200"], color=COLORS["cyan"],
             linewidth=0.8, alpha=0.7, label="SMA 200")
    ax1.fill_between(dates, df["BB_upper"], df["BB_lower"],
                     alpha=0.07, color=COLORS["blue"])

    # Ichimoku cloud (recent 500 bars for clarity)
    recent_n = min(500, len(dates))
    ax1.fill_between(dates[-recent_n:],
                     df["Ichi_span_a"].iloc[-recent_n:],
                     df["Ichi_span_b"].iloc[-recent_n:],
                     where=df["Ichi_span_a"].iloc[-recent_n:] >= df["Ichi_span_b"].iloc[-recent_n:],
                     alpha=0.08, color=COLORS["green"], label="Ichi Cloud (Bull)")
    ax1.fill_between(dates[-recent_n:],
                     df["Ichi_span_a"].iloc[-recent_n:],
                     df["Ichi_span_b"].iloc[-recent_n:],
                     where=df["Ichi_span_a"].iloc[-recent_n:] < df["Ichi_span_b"].iloc[-recent_n:],
                     alpha=0.08, color=COLORS["red"], label="Ichi Cloud (Bear)")

    for s in c["all_supports"][-3:]:
        ax1.axhline(s, color=COLORS["green"], linewidth=0.7,
                    linestyle="--", alpha=0.5)
    for r in c["all_resistances"][:3]:
        ax1.axhline(r, color=COLORS["red"], linewidth=0.7,
                    linestyle="--", alpha=0.5)

    ax1.axvline(val_d[0], color=COLORS["yellow"],
                linewidth=1.5, linestyle=":", alpha=0.8, label="Train/Val split")

    last_date  = dates[-1]
    fut_date   = c["est_end"]
    lp         = c["last_price"]
    pt         = c["price_target"]
    arrow_col  = COLORS["green"] if c["trend"] == "BULLISH" else COLORS["red"]

    ax1.annotate("", xy=(fut_date, pt), xytext=(last_date, lp),
                 arrowprops=dict(arrowstyle="->", color=arrow_col,
                                 lw=2.5, connectionstyle="arc3,rad=0.15"))
    ax1.scatter([fut_date], [pt], color=arrow_col, s=100, zorder=6)
    ax1.text(fut_date, pt, f"  Target\n  ${pt:.2f}",
             color=arrow_col, fontsize=8, fontweight="bold", va="center")

    ax1.legend(loc="upper left", fontsize=7, framealpha=0.2, labelcolor=COLORS["text"])
    ax_style(ax1,
             f"{TICKER} — Full History | BB Bands | Ichimoku Cloud | Trend Projection",
             "Price (USD)")

    # ─────────────────────────────────────────
    # CHART 2 — Volume  (row 1, col 0)
    # ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    colors_vol = [COLORS["green"] if c_ >= o else COLORS["red"]
                  for c_, o in zip(df["Close"], df["Open"])]
    ax2.bar(dates, df["Volume"], color=colors_vol, alpha=0.7, width=1)
    ax2.plot(dates, df["Vol_SMA20"], color=COLORS["yellow"], linewidth=1, label="Vol SMA20")
    ax_style(ax2, "Volume", "Shares")
    ax2.legend(fontsize=7, framealpha=0.1)

    # ─────────────────────────────────────────
    # CHART 3 — RSI  (row 1, col 1)
    # ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(dates, df["RSI"], color=COLORS["purple"], linewidth=1)
    ax3.axhline(70, color=COLORS["red"],   linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.axhline(30, color=COLORS["green"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.fill_between(dates, df["RSI"], 70,
                     where=df["RSI"] > 70, alpha=0.15, color=COLORS["red"])
    ax3.fill_between(dates, df["RSI"], 30,
                     where=df["RSI"] < 30, alpha=0.15, color=COLORS["green"])
    ax3.set_ylim(0, 100)
    ax3.text(dates[-1], 72, "OB", color=COLORS["red"],   fontsize=7)
    ax3.text(dates[-1], 26, "OS", color=COLORS["green"], fontsize=7)
    ax_style(ax3, "RSI (14)", "RSI")

    # ─────────────────────────────────────────
    # CHART 4 — MACD  (row 1, col 2)
    # ─────────────────────────────────────────
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

    # ─────────────────────────────────────────
    # CHART 5 — Validation actual vs predicted  (row 2, col 0-1)
    # ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    val_close = df.loc[val_d, "Close"].values
    ax5.plot(val_d, val_close, color=COLORS["blue"],
             linewidth=1.2, label="Actual price", alpha=0.9)
    for i in range(len(val_d)):
        correct = (ensemble_pred[i] == y_val[i])
        col = COLORS["green"] if correct else COLORS["red"]
        ax5.axvspan(val_d[i], val_d[min(i+1, len(val_d)-1)],
                    alpha=0.12, color=col)
    ax5.set_title("Validation Period — Prediction Accuracy (Green=Correct, Red=Wrong)",
                  color=COLORS["text"], fontsize=9, fontweight="bold", pad=6)
    ax_style(ax5, "", "Price (USD)")
    ax5b = ax5.twinx()
    ax5b.plot(val_d, ensemble_proba, color=COLORS["purple"],
              linewidth=0.9, alpha=0.7, label="Bullish prob")
    ax5b.axhline(0.5, color=COLORS["yellow"],
                 linewidth=0.6, linestyle="--", alpha=0.6)
    ax5b.set_ylim(0, 1)
    ax5b.set_ylabel("Bullish probability", color=COLORS["purple"], fontsize=8)
    ax5b.tick_params(colors=COLORS["purple"])
    lines1, labs1 = ax5.get_legend_handles_labels()
    lines2, labs2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labs1 + labs2,
               fontsize=7, framealpha=0.15, loc="upper left")

    # ─────────────────────────────────────────
    # CHART 6 — Model metrics bar  (row 2, col 2)
    # ─────────────────────────────────────────
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

    # ─────────────────────────────────────────
    # CHART 7 — Z-SCORE COMPOSITE  (row 3, col 0-1)
    # NEW: Rolling Z-score composite timeseries
    # ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, :2])
    recent_n2 = min(500, len(dates))
    z_vals  = df["Z_composite"].iloc[-recent_n2:]
    z_dates = dates[-recent_n2:]
    ax7.plot(z_dates, z_vals, color=COLORS["cyan"], linewidth=1.0, label="Z Composite")
    ax7.fill_between(z_dates, z_vals, 0,
                     where=z_vals > 0, alpha=0.18, color=COLORS["green"])
    ax7.fill_between(z_dates, z_vals, 0,
                     where=z_vals < 0, alpha=0.18, color=COLORS["red"])
    ax7.axhline(Z_EXTREME,  color=COLORS["red"],    linewidth=0.8, linestyle="--", alpha=0.7, label=f"±{Z_EXTREME} (overextended)")
    ax7.axhline(-Z_EXTREME, color=COLORS["red"],    linewidth=0.8, linestyle="--", alpha=0.7)
    ax7.axhline(Z_MODERATE,  color=COLORS["yellow"], linewidth=0.7, linestyle=":", alpha=0.6, label=f"±{Z_MODERATE} (moderate)")
    ax7.axhline(-Z_MODERATE, color=COLORS["yellow"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax7.axhline(0, color=COLORS["muted"], linewidth=0.5)

    # Mark current Z on right edge
    curr_z = c["z_composite"]
    z_col  = COLORS["red"] if abs(curr_z) > Z_EXTREME else COLORS["yellow"] if abs(curr_z) > Z_MODERATE else COLORS["green"]
    ax7.axhline(curr_z, color=z_col, linewidth=1.0, linestyle="-", alpha=0.5)
    ax7.text(z_dates[-1], curr_z, f"  {curr_z:+.2f}", color=z_col, fontsize=8, fontweight="bold", va="center")

    ax7.legend(fontsize=7, framealpha=0.1, loc="upper left")
    ax_style(ax7, "Z-Score Composite (last 500 bars) — Overextension Indicator", "Z-Score")

    # ─────────────────────────────────────────
    # CHART 8 — Hurst Exponent  (row 3, col 2)
    # ─────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 2])
    hurst_vals  = df["Hurst"].dropna()
    hurst_dates = hurst_vals.index
    ax8.plot(hurst_dates, hurst_vals, color=COLORS["orange"], linewidth=1.0, label="Hurst (100d)")
    ax8.fill_between(hurst_dates, hurst_vals, 0.5,
                     where=hurst_vals > 0.5, alpha=0.2, color=COLORS["green"], label="Trending")
    ax8.fill_between(hurst_dates, hurst_vals, 0.5,
                     where=hurst_vals < 0.5, alpha=0.2, color=COLORS["red"], label="Mean-Rev")
    ax8.axhline(0.55, color=COLORS["green"],  linewidth=0.8, linestyle="--", alpha=0.6)
    ax8.axhline(0.50, color=COLORS["muted"],  linewidth=0.8, linestyle="-",  alpha=0.5)
    ax8.axhline(0.45, color=COLORS["red"],    linewidth=0.8, linestyle="--", alpha=0.6)
    curr_h = c["hurst"]
    ax8.scatter([hurst_dates[-1]], [curr_h], color=COLORS["cyan"], s=60, zorder=5)
    ax8.text(hurst_dates[-1], curr_h + 0.01, f"  {curr_h:.3f}", color=COLORS["cyan"], fontsize=8)
    ax8.set_ylim(0.2, 0.9)
    ax8.legend(fontsize=7, framealpha=0.1)
    ax_style(ax8, f"Hurst Exponent — Regime: {c['regime']}", "H")

    # ─────────────────────────────────────────
    # CHART 9 — Feature importance  (row 4, col 0-1)
    # ─────────────────────────────────────────
    ax9 = fig.add_subplot(gs[4, :2])
    feats = [f[0] for f in top_features][::-1]
    imps  = [f[1] for f in top_features][::-1]
    # Highlight Z-score features
    bar_colors = []
    for feat in feats:
        if feat.startswith("Z_") or feat == "Hurst":
            bar_colors.append(COLORS["cyan"])
        elif feat.startswith("Ichi"):
            bar_colors.append(COLORS["green"])
        else:
            bar_colors.append(COLORS["blue"])
    bars = ax9.barh(feats, imps, color=bar_colors, alpha=0.8, height=0.7)
    for bar, val in zip(bars, imps):
        ax9.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=7, color=COLORS["muted"])

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=COLORS["cyan"],  label="Z-Score / Hurst (new)"),
        Patch(facecolor=COLORS["green"], label="Ichimoku (new)"),
        Patch(facecolor=COLORS["blue"],  label="Classic indicators"),
    ]
    ax9.legend(handles=legend_els, fontsize=7, framealpha=0.15, loc="lower right")
    ax_style(ax9, "Top 15 Feature Importances (XGBoost) — Cyan=Z/Hurst, Green=Ichimoku", "Importance")

    # ─────────────────────────────────────────
    # CHART 10 — ADX  (row 4, col 2)
    # ─────────────────────────────────────────
    ax10 = fig.add_subplot(gs[4, 2])
    ax10.plot(dates, df["ADX"],     color=COLORS["orange"], linewidth=1, label="ADX")
    ax10.plot(dates, df["ADX_pos"], color=COLORS["green"],  linewidth=0.8, alpha=0.7, label="+DI")
    ax10.plot(dates, df["ADX_neg"], color=COLORS["red"],    linewidth=0.8, alpha=0.7, label="-DI")
    ax10.axhline(25, color=COLORS["yellow"],
                linewidth=0.7, linestyle="--", alpha=0.6, label="Trend threshold")
    ax10.legend(fontsize=7, framealpha=0.1)
    ax_style(ax10, "ADX — Trend Strength", "")

    # ─────────────────────────────────────────
    # CHART 11 — Prediction summary panel (row 5, full)
    # ─────────────────────────────────────────
    ax11 = fig.add_subplot(gs[5, :])
    ax11.set_facecolor(COLORS["panel"])
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.axis("off")

    rect = FancyBboxPatch((0.01, 0.03), 0.98, 0.94,
                           boxstyle="round,pad=0.01",
                           linewidth=1.5,
                           edgecolor=(COLORS["green"] if c["trend"] == "BULLISH"
                                      else COLORS["red"]),
                           facecolor=COLORS["bg"], alpha=0.8)
    ax11.add_patch(rect)

    trend_col = COLORS["green"] if c["trend"] == "BULLISH" else COLORS["red"]
    trend_sym  = "▲" if c["trend"] == "BULLISH" else "▼"

    ax11.text(0.5, 0.91,
              f"{trend_sym}  {c['ticker']} — {c['trend']} FOR NEXT ~{c['est_days']} TRADING DAYS  {trend_sym}",
              ha="center", va="center", fontsize=15, fontweight="bold",
              color=trend_col, transform=ax11.transAxes)

    # Regime badge
    regime_col = COLORS["cyan"] if c["regime"] == "TRENDING" else COLORS["orange"] if c["regime"] == "MEAN-REVERTING" else COLORS["muted"]
    ax11.text(0.5, 0.82,
              f"Market Regime: {c['regime']}  (Hurst={c['hurst']:.3f})  |  "
              f"Z-Composite: {c['z_composite']:+.2f}  |  "
              f"Conf adj: raw {c['raw_confidence']*100:.1f}% → {c['confidence']*100:.1f}%",
              ha="center", va="center", fontsize=8.5,
              color=regime_col, transform=ax11.transAxes)

    # key stats — 3 columns
    col_x = [0.12, 0.45, 0.78]
    rows = [
        [
            f"Current Price   ${c['last_price']:.2f}",
            f"Price Target    ${c['price_target']:.2f}",
            f"Est Duration    ~{c['est_days']} trading days",
        ],
        [
            f"Confidence      {c['confidence']*100:.1f}%",
            f"Stop Loss       ${c['stop_loss']:.2f}",
            f"Target Date     {c['est_end'].date()}",
        ],
        [
            f"ATR             ${c['atr']:.2f}",
            f"ADX             {c['adx']:.1f} ({'Strong' if c['adx']>25 else 'Mod'})",
            f"Hist Vol        {c['hist_vol']*100:.1f}% p.a.",
        ],
        [
            f"Z_price_20      {c['z_price_20']:+.2f}",
            f"Z_price_50      {c['z_price_50']:+.2f}",
            f"Z_composite     {c['z_composite']:+.2f}",
        ],
        [
            f"Support         ${c['near_sup']:.2f}" if c['near_sup'] else "Support    —",
            f"Resistance      ${c['near_res']:.2f}" if c['near_res'] else "Resistance —",
            f"Move            {(c['price_target']-c['last_price'])/c['last_price']*100:+.1f}%",
        ],
    ]

    y_positions = [0.70, 0.58, 0.46, 0.34, 0.22]
    for row_y, row in zip(y_positions, rows):
        for cx, txt in zip(col_x, row):
            label, _, val = txt.partition("  ")
            ax11.text(cx - 0.06, row_y, label,
                      ha="left", fontsize=8.5, color=COLORS["muted"],
                      transform=ax11.transAxes)
            ax11.text(cx + 0.08, row_y, val.strip(),
                      ha="left", fontsize=9, fontweight="bold",
                      color=COLORS["text"], transform=ax11.transAxes)

    ax11.text(0.5, 0.08,
              "⚠  For research & educational purposes only. Not financial advice.",
              ha="center", fontsize=7.5, color=COLORS["muted"],
              transform=ax11.transAxes, style="italic")

    fig.suptitle(
        f"QUANTITATIVE TREND PREDICTOR v2.0  |  {TICKER}  |  "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=11, fontweight="bold",
        color=COLORS["text"], y=0.972)

    import os
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/{TICKER}_quant_predictor_v2.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✓  Chart saved → {out}")
    return out


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main(ticker: str = TICKER):
    global TICKER
    TICKER = ticker.upper()

    df         = fetch_data(TICKER)
    df         = add_features(df)

    (models, scaler, feature_cols,
     df, split, dates_val,
     preds_val, probas_val, ensemble_pred, ensemble_proba,
     y_val, metrics, top_features) = train_and_validate(df)

    conclusion = make_prediction(models, scaler, feature_cols,
                                 df, ensemble_proba, y_val)

    print_conclusion(conclusion, metrics)

    out = plot_all(df, split, dates_val,
                   ensemble_pred, ensemble_proba, y_val,
                   metrics, top_features, conclusion)

    print(f"\n  Done!  Output → {out}\n")
    return conclusion


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else TICKER
    main(ticker)
