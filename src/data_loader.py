from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .utils import PROJECT_ROOT


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class MarketDataUnavailableError(ValueError):
    """Raised when required market history cannot be retrieved."""


@dataclass
class MarketDataBundle:
    ticker: str
    ticker_data: pd.DataFrame
    benchmark_primary: str
    benchmark_primary_data: pd.DataFrame
    benchmark_secondary: str
    benchmark_secondary_data: pd.DataFrame
    vix_data: pd.DataFrame | None = None
    treasury_yield_ticker: str = "^TNX"
    treasury_yield_data: pd.DataFrame | None = None
    oil_ticker: str = "CL=F"
    oil_data: pd.DataFrame | None = None
    warnings: list[str] | None = None


def _normalize_yfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.copy()
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required fields: {missing}")
    df = df[REQUIRED_COLUMNS].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def _download_yahoo_chart(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    final_date = date.fromisoformat(end) if end else date.today() + timedelta(days=1)
    period2 = int(pd.Timestamp(final_date, tz="UTC").timestamp())
    params = urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}?{params}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 QuantMarketResearch/2.1"})

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise MarketDataUnavailableError(f"Yahoo chart request failed for {ticker}: {exc}") from exc

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise MarketDataUnavailableError(f"Yahoo chart request failed for {ticker}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise MarketDataUnavailableError(f"Yahoo chart returned no rows for {ticker}.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_rows = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjclose_rows = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
    if not timestamps:
        raise MarketDataUnavailableError(f"Yahoo chart returned no rows for {ticker}.")

    close = quote_rows.get("close", [])
    adjusted_close = adjclose_rows.get("adjclose") or close
    frame = pd.DataFrame(
        {
            "Open": quote_rows.get("open", []),
            "High": quote_rows.get("high", []),
            "Low": quote_rows.get("low", []),
            "Close": close,
            "Adj Close": adjusted_close,
            "Volume": quote_rows.get("volume", []),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
    )
    return _normalize_yfinance_frame(frame)


def download_ohlcv(ticker: str, start: str, end: str | None = None, cache: bool = True) -> pd.DataFrame:
    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("^", "INDEX_").replace("/", "_")
    end_part = end or str(date.today())
    cache_path = cache_dir / f"{safe_ticker}_{start}_{end_part}.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col="Date", parse_dates=True)

    try:
        normalized = _download_yahoo_chart(ticker, start, end)
    except MarketDataUnavailableError as chart_exc:
        try:
            import yfinance as yf

            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
            if df.empty:
                raise MarketDataUnavailableError(f"yfinance returned no rows for {ticker}.")
            normalized = _normalize_yfinance_frame(df)
        except Exception as yfinance_exc:
            raise MarketDataUnavailableError(
                f"Unable to download OHLCV data for {ticker}. "
                f"{chart_exc}; yfinance fallback failed ({yfinance_exc}). "
                "The market data provider may be temporarily unavailable from this server."
            ) from yfinance_exc
    if cache:
        normalized.to_csv(cache_path)
    return normalized


def load_market_data(
    ticker: str,
    start: str,
    end: str | None = None,
    benchmark_primary: str = "QQQ",
    benchmark_secondary: str = "SPY",
    treasury_yield_ticker: str = "^TNX",
    oil_ticker: str = "CL=F",
) -> MarketDataBundle:
    warnings: list[str] = []
    requests = {
        "ticker": ticker,
        "primary": benchmark_primary,
        "secondary": benchmark_secondary,
        "vix": "^VIX",
        "treasury": treasury_yield_ticker,
        "oil": oil_ticker,
    }
    results: dict[str, pd.DataFrame | None] = {}
    errors: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = {
            executor.submit(download_ohlcv, symbol, start, end): name for name, symbol in requests.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                errors[name] = exc
                results[name] = None

    for name in ("ticker", "primary", "secondary"):
        if name in errors:
            raise errors[name]
    if "vix" in errors:
        warnings.append(f"VIX data unavailable: {errors['vix']}")
    if "treasury" in errors:
        warnings.append(f"Treasury yield data unavailable for {treasury_yield_ticker}: {errors['treasury']}")
    if "oil" in errors:
        warnings.append(f"Oil data unavailable for {oil_ticker}: {errors['oil']}")

    return MarketDataBundle(
        ticker=ticker.upper(),
        ticker_data=results["ticker"],
        benchmark_primary=benchmark_primary.upper(),
        benchmark_primary_data=results["primary"],
        benchmark_secondary=benchmark_secondary.upper(),
        benchmark_secondary_data=results["secondary"],
        vix_data=results["vix"],
        treasury_yield_ticker=treasury_yield_ticker.upper(),
        treasury_yield_data=results["treasury"],
        oil_ticker=oil_ticker.upper(),
        oil_data=results["oil"],
        warnings=warnings,
    )
