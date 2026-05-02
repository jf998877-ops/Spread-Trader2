"""
Credit Spread Scanner + Paper Trader — Streamlit Web App
=========================================================
Combines the spread scanner and paper trade tracker into a web interface.

To run locally:
    pip install streamlit yfinance pandas numpy scipy requests
    streamlit run app.py

To deploy free at <yourapp>.streamlit.app:
    1. Push this file (and requirements.txt) to a public GitHub repo
    2. Go to https://share.streamlit.io
    3. Sign in with GitHub, click "New app", point at the repo
    4. Set main file path to app.py
    5. Configure persistent storage (see below)

PERSISTENT STORAGE — IMPORTANT
==============================
Streamlit Cloud has an ephemeral filesystem — local JSON files are wiped on
every restart. To persist trades long-term, configure GitHub-backed storage:

  1. Create a GitHub Personal Access Token (PAT) at:
     https://github.com/settings/tokens?type=beta
     - Click "Generate new token" → "Fine-grained token"
     - Repository access: select the GitHub repo where you'll store trades
       (can be the same repo as the app, or a separate private repo)
     - Permissions: Contents → Read and write
     - Generate, copy the token (you'll only see it once)

  2. In your Streamlit Cloud app dashboard:
     - Click "Settings" → "Secrets"
     - Paste the following (with your values):

         GITHUB_TOKEN = "ghp_yourtokenhere"
         GITHUB_REPO = "yourusername/your-repo-name"
         GITHUB_FILE = "trades.json"
         GITHUB_BRANCH = "main"

  3. Save secrets. The app will detect them and use GitHub for storage.

Without these secrets, the app falls back to local file storage which is
ephemeral on Streamlit Cloud (your trades will reset on restart).

requirements.txt should contain:
    streamlit
    yfinance
    pandas
    numpy
    scipy
    requests
"""

import base64
import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.045

DEFAULT_TICKERS = [
    "BAC", "JPM", "WFC", "C", "GS", "MS", "SCHW",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "KO", "PEP", "PG", "WMT", "COST", "MCD",
    "XOM", "CVX", "COP",
    "JNJ", "PFE", "UNH", "ABBV",
    "CAT", "BA", "GE", "DIS", "F", "T", "VZ",
    "SPY", "QQQ", "IWM", "XLF", "XLE",
]

# Local filesystem fallback — used when GitHub secrets aren't configured.
# On Streamlit Cloud this is ephemeral; configure GitHub storage for persistence.
LOCAL_STORAGE = Path.home() / ".spread_paper_trades.json"


# ---------------------------------------------------------------------------
# Storage backend — GitHub API (primary) or local file (fallback)
# ---------------------------------------------------------------------------
def _get_github_config() -> dict | None:
    """Return GitHub config dict from secrets, or None if not configured."""
    try:
        token = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("GITHUB_REPO")
        if not token or not repo:
            return None
        return {
            "token": token,
            "repo": repo,
            "file": st.secrets.get("GITHUB_FILE", "trades.json"),
            "branch": st.secrets.get("GITHUB_BRANCH", "main"),
        }
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def _github_url(cfg: dict) -> str:
    return f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['file']}"


def _github_headers(cfg: dict) -> dict:
    return {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _github_load() -> tuple[list[dict], str | None]:
    """
    Load JSON from GitHub. Returns (data_list, file_sha).
    file_sha is needed to write updates (GitHub requires it for PUT).
    Returns ([], None) if file doesn't exist yet.
    Raises on auth/network errors.
    """
    cfg = _get_github_config()
    if not cfg:
        raise RuntimeError("GitHub not configured")

    r = requests.get(
        _github_url(cfg),
        headers=_github_headers(cfg),
        params={"ref": cfg["branch"]},
        timeout=10,
    )
    if r.status_code == 404:
        return [], None  # File doesn't exist yet — first save will create it
    r.raise_for_status()
    payload = r.json()
    content = base64.b64decode(payload["content"]).decode("utf-8")
    if not content.strip():
        return [], payload["sha"]
    return json.loads(content), payload["sha"]


def _github_save(data: list[dict]) -> None:
    """Save JSON to GitHub, creating or updating the file with a commit."""
    cfg = _get_github_config()
    if not cfg:
        raise RuntimeError("GitHub not configured")

    # Need the current SHA to update an existing file
    try:
        _, sha = _github_load()
    except requests.HTTPError:
        sha = None

    body = json.dumps(data, indent=2)
    encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")

    payload = {
        "message": f"Update trades ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "content": encoded,
        "branch": cfg["branch"],
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(
        _github_url(cfg),
        headers=_github_headers(cfg),
        json=payload,
        timeout=15,
    )
    r.raise_for_status()


def storage_backend_name() -> str:
    """Return a human-readable name of the active storage backend."""
    cfg = _get_github_config()
    if cfg:
        return f"GitHub ({cfg['repo']}/{cfg['file']})"
    return f"Local file ({LOCAL_STORAGE}) — EPHEMERAL on Streamlit Cloud!"


def is_github_configured() -> bool:
    return _get_github_config() is not None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class PaperTrade:
    trade_id: str
    ticker: str
    spread_type: str
    short_strike: float
    long_strike: float
    expiration: str
    contracts: int
    entry_date: str
    entry_price: float
    entry_credit: float
    max_profit: float
    max_loss: float
    breakeven: float
    notes: str = ""
    status: str = "OPEN"
    close_date: str = ""
    close_debit: float = 0.0
    realized_pnl: float = 0.0
    close_reason: str = ""
    source: str = "MANUAL"  # MANUAL or AUTO (from high-score scanner result)
    scanner_score: int = 0  # Score from scanner at time of entry (0 if manual)


def load_trades() -> list[PaperTrade]:
    """
    Load trades from the active storage backend.
    Uses GitHub if configured, otherwise local file.
    Caches in st.session_state to avoid hitting GitHub on every rerun.
    """
    # Use a session-state cache to avoid GitHub API calls on every Streamlit rerun.
    # The cache is invalidated on every save_trades() call.
    if "trades_cache" in st.session_state:
        return st.session_state["trades_cache"]

    raw_data: list[dict] = []
    if is_github_configured():
        try:
            raw_data, _ = _github_load()
        except Exception as e:
            st.error(f"GitHub load failed: {e}. Falling back to local file.")
            if LOCAL_STORAGE.exists():
                try:
                    with open(LOCAL_STORAGE, "r") as f:
                        raw_data = json.load(f)
                except Exception:
                    raw_data = []
    else:
        if LOCAL_STORAGE.exists():
            try:
                with open(LOCAL_STORAGE, "r") as f:
                    raw_data = json.load(f)
            except Exception:
                raw_data = []

    # Be backward compatible with older JSON files
    trades = []
    for t in raw_data:
        t.setdefault("source", "MANUAL")
        t.setdefault("scanner_score", 0)
        try:
            trades.append(PaperTrade(**t))
        except TypeError:
            # Skip records with unknown fields rather than crashing
            continue

    st.session_state["trades_cache"] = trades
    return trades


def save_trades(trades: list[PaperTrade]) -> None:
    """Save trades to the active storage backend and refresh the cache."""
    data = [asdict(t) for t in trades]

    if is_github_configured():
        try:
            _github_save(data)
        except Exception as e:
            st.error(f"GitHub save failed: {e}. Saving to local file as backup.")
            LOCAL_STORAGE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOCAL_STORAGE, "w") as f:
                json.dump(data, f, indent=2)
    else:
        LOCAL_STORAGE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCAL_STORAGE, "w") as f:
            json.dump(data, f, indent=2)

    # Refresh the cache with the new data
    st.session_state["trades_cache"] = trades


# ---------------------------------------------------------------------------
# Indicator math (same as the CLI scripts)
# ---------------------------------------------------------------------------
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_hv_rank(close: pd.Series) -> float:
    log_returns = np.log(close / close.shift(1))
    rolling_hv = log_returns.rolling(window=30).std() * np.sqrt(252) * 100
    recent = rolling_hv.dropna().tail(252)
    if len(recent) < 20:
        return float("nan")
    current = recent.iloc[-1]
    hv_min, hv_max = recent.min(), recent.max()
    if hv_max == hv_min:
        return 50.0
    return float((current - hv_min) / (hv_max - hv_min) * 100)


def calc_30d_hv(close: pd.Series) -> float:
    log_returns = np.log(close / close.shift(1))
    return float(log_returns.tail(30).std() * np.sqrt(252))


def consecutive_streak(close: pd.Series, direction: str = "down") -> int:
    diffs = close.diff().dropna()
    streak = 0
    for d in diffs.iloc[::-1]:
        if (direction == "down" and d < 0) or (direction == "up" and d > 0):
            streak += 1
        else:
            break
    return streak


def calc_volume_ratio(volume: pd.Series, period: int = 20) -> float:
    """
    Today's volume vs the recent average.
    >1.0 = above average, <1.0 = below average.
    A pullback on volume_ratio < 0.8 is healthy.
    A pullback on volume_ratio > 1.5 suggests distribution.
    """
    if len(volume) < period + 1:
        return float("nan")
    today = float(volume.iloc[-1])
    avg = float(volume.iloc[-(period + 1):-1].mean())  # exclude today from avg
    if avg == 0:
        return float("nan")
    return today / avg


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """
    Average Directional Index — trend STRENGTH (not direction).
    < 20: choppy, no real trend
    20-25: developing trend
    25-40: strong trend
    > 40: very strong trend (often near exhaustion)
    Wilder's original method using smoothed averages.
    """
    if len(close) < period * 2 + 1:
        return float("nan")

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    # Smoothed using Wilder's method (EMA equivalent)
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return float(adx.iloc[-1]) if not adx.empty else float("nan")


def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int = 14) -> float:
    """
    Money Flow Index — volume-weighted RSI.
    < 20: oversold + capitulation volume (strong reversal candidate)
    > 80: overbought + distribution volume (good for bear setups)
    Better than plain RSI for confirming whether moves have institutional weight.
    """
    if len(close) < period + 1:
        return float("nan")
    typical = (high + low + close) / 3
    money_flow = typical * volume
    delta = typical.diff()
    pos_flow = money_flow.where(delta > 0, 0.0)
    neg_flow = money_flow.where(delta < 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return float(mfi.iloc[-1]) if not mfi.dropna().empty else float("nan")


def calc_zscore(close: pd.Series, period: int = 20) -> float:
    """
    Z-score of current price vs its rolling mean.
    < -2: statistically oversold (good bull put zone)
    -2 to -1: pulled back
    -1 to +1: normal range
    +1 to +2: extended
    > +2: statistically overbought (good bear call zone)
    """
    if len(close) < period:
        return float("nan")
    window = close.tail(period)
    mean = window.mean()
    std = window.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((close.iloc[-1] - mean) / std)


def calc_vwap_distance(high: pd.Series, low: pd.Series, close: pd.Series,
                        volume: pd.Series, period: int = 20) -> float:
    """
    Distance from rolling VWAP, expressed as percent.
    Negative = price below VWAP (institutions underwater on average).
    Positive = price above VWAP.
    Mean-reversion to VWAP is a strong tendency.
    Strong support typically forms when price reverts back up to VWAP from below.

    Note: true daily VWAP resets each session. This is a 20-day rolling
    approximation, which is what's possible with daily bars.
    """
    if len(close) < period:
        return float("nan")
    typical = (high + low + close) / 3
    vol_window = volume.tail(period)
    typ_window = typical.tail(period)
    total_vol = vol_window.sum()
    if total_vol == 0:
        return float("nan")
    vwap = (typ_window * vol_window).sum() / total_vol
    return float((close.iloc[-1] - vwap) / vwap * 100)


def bs_price(S: float, K: float, T: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return float(max(S - K, 0) if is_call else max(K - S, 0))
    d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return float(S * norm.cdf(d1) - K * np.exp(-RISK_FREE_RATE * T) * norm.cdf(d2))
    return float(K * np.exp(-RISK_FREE_RATE * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def prob_above(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d2 = (np.log(S / K) + (RISK_FREE_RATE - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(1.0 - norm.cdf(-d2))


def prob_below(S: float, K: float, T: float, sigma: float) -> float:
    return 1.0 - prob_above(S, K, T, sigma)


def round_strike(x: float) -> float:
    if x < 25:
        return round(x * 2) / 2
    if x < 200:
        return float(round(x))
    return round(x / 5) * 5


def strike_width(price: float) -> float:
    if price < 25:
        return 1.0
    if price < 75:
        return 2.5
    if price < 200:
        return 5.0
    return 10.0


def next_monthly_expiration(target_dte: int = 30) -> datetime:
    today = datetime.now()
    target = today + timedelta(days=target_dte)
    year, month = target.year, target.month
    first_day = datetime(year, month, 1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    third_friday = first_day + timedelta(days=first_friday_offset + 14)
    # Always require at least 30 DTE — never return an expiration closer than that
    min_dte = max(30, target_dte // 2)
    if (third_friday - today).days < min_dte:
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        first_day = datetime(year, month, 1)
        first_friday_offset = (4 - first_day.weekday()) % 7
        third_friday = first_day + timedelta(days=first_friday_offset + 14)
    return third_friday


# ---------------------------------------------------------------------------
# Data fetch — cached so reruns don't hammer Yahoo
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # 5 minute cache
def fetch_history(ticker: str, days: int = 420) -> pd.DataFrame | None:
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        hist = yf.download(ticker, start=start, end=end,
                           progress=False, auto_adjust=True, threads=False)
        if hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_price_and_vol(ticker: str) -> tuple[float, float] | None:
    hist = fetch_history(ticker, days=90)
    if hist is None or len(hist) < 20:
        return None
    close = hist["Close"].dropna()
    price = float(close.iloc[-1])
    log_returns = np.log(close / close.shift(1)).dropna()
    sigma = float(log_returns.tail(30).std() * np.sqrt(252))
    return price, sigma


# ---------------------------------------------------------------------------
# Scanner core
# ---------------------------------------------------------------------------
def analyze_ticker(ticker: str) -> dict | None:
    hist = fetch_history(ticker, days=420)
    if hist is None or len(hist) < 210:
        return None
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    high = hist["High"].dropna()
    low = hist["Low"].dropna()

    price = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    rsi14 = float(calc_rsi(close, 14).iloc[-1])
    hv_rank = calc_hv_rank(close)
    sigma = calc_30d_hv(close)
    window60 = close.tail(60)
    support, resistance = float(window60.min()), float(window60.max())
    avg_vol = float(volume.tail(20).mean())
    down_streak = consecutive_streak(close, "down")
    up_streak = consecutive_streak(close, "up")

    # New indicators
    volume_ratio = calc_volume_ratio(volume, period=20)
    adx = calc_adx(high, low, close, period=14)
    mfi = calc_mfi(high, low, close, volume, period=14)
    zscore = calc_zscore(close, period=20)
    vwap_dist = calc_vwap_distance(high, low, close, volume, period=20)

    return {
        "Ticker": ticker, "Price": price,
        "SMA50": sma50, "SMA200": sma200,
        "RSI14": rsi14, "HV_Rank": hv_rank, "Sigma": sigma,
        "Support": support, "Resistance": resistance,
        "Dist_Support_%": (price - support) / support * 100,
        "Dist_Resistance_%": (price - resistance) / resistance * 100,
        "Above_200SMA": price > sma200,
        "Above_50SMA": price > sma50,
        "Avg_Volume_20d": avg_vol,
        "Down_Streak": down_streak,
        "Up_Streak": up_streak,
        # New
        "Volume_Ratio": volume_ratio,
        "ADX": adx,
        "MFI": mfi,
        "ZScore": zscore,
        "VWAP_Dist_%": vwap_dist,
    }


def classify(row: dict) -> tuple[str, int, list[str]]:
    if row["Avg_Volume_20d"] < 1_000_000:
        return "AVOID", 0, ["Insufficient liquidity"]

    rsi, above_200 = row["RSI14"], row["Above_200SMA"]
    above_50, hv_rank = row["Above_50SMA"], row["HV_Rank"]
    dist_s, dist_r = row["Dist_Support_%"], row["Dist_Resistance_%"]
    down, up = row["Down_Streak"], row["Up_Streak"]
    vol_ratio = row.get("Volume_Ratio", float("nan"))
    adx = row.get("ADX", float("nan"))
    mfi = row.get("MFI", float("nan"))
    zscore = row.get("ZScore", float("nan"))
    vwap_dist = row.get("VWAP_Dist_%", float("nan"))

    # ------ BULL PUT SCORING ------
    bull, bull_r = 0, []
    if above_200: bull += 20; bull_r.append("Above 200-SMA")
    if rsi < 35: bull += 20; bull_r.append(f"RSI {rsi:.1f} oversold")
    elif rsi < 45: bull += 10; bull_r.append(f"RSI {rsi:.1f} pulled back")
    if -2 < dist_s < 5: bull += 15; bull_r.append(f"Near support ({dist_s:+.1f}%)")
    if not np.isnan(hv_rank):
        if hv_rank > 50: bull += 15; bull_r.append(f"HV Rank {hv_rank:.0f}")
        elif hv_rank > 30: bull += 8; bull_r.append(f"HV Rank {hv_rank:.0f}")
    if above_50 and above_200: bull += 8; bull_r.append("Above both MAs")
    if down >= 2: bull += min(12, 4 * down); bull_r.append(f"{down}d down streak")

    # MFI confirmation — oversold w/ volume = capitulation = strong bull put setup
    if not np.isnan(mfi):
        if mfi < 20: bull += 15; bull_r.append(f"MFI {mfi:.0f} oversold (capitulation)")
        elif mfi < 30: bull += 8; bull_r.append(f"MFI {mfi:.0f} weak")

    # Z-score: statistically oversold = mean reversion candidate
    if not np.isnan(zscore):
        if zscore < -2: bull += 15; bull_r.append(f"Z-score {zscore:.1f} (extreme oversold)")
        elif zscore < -1: bull += 8; bull_r.append(f"Z-score {zscore:.1f} (oversold)")

    # ADX — trend strength filter (we want SOME trend, but not parabolic)
    if not np.isnan(adx):
        if 20 <= adx <= 40: bull += 8; bull_r.append(f"ADX {adx:.0f} (healthy trend)")
        elif adx > 40: bull -= 5; bull_r.append(f"ADX {adx:.0f} (overextended trend)")
        elif adx < 15: bull -= 3; bull_r.append(f"ADX {adx:.0f} (choppy/no trend)")

    # Volume ratio on a pullback — low volume = healthy pullback (good for bull put)
    if not np.isnan(vol_ratio) and down >= 1:
        if vol_ratio < 0.8: bull += 10; bull_r.append(f"Pullback on weak volume ({vol_ratio:.1f}x)")
        elif vol_ratio > 1.5: bull -= 8; bull_r.append(f"Pullback on heavy volume ({vol_ratio:.1f}x — distribution?)")

    # VWAP distance — being well below VWAP = stretched, often bounces back
    if not np.isnan(vwap_dist):
        if vwap_dist < -3: bull += 8; bull_r.append(f"{vwap_dist:.1f}% below VWAP")
        elif vwap_dist < -1.5: bull += 4; bull_r.append(f"{vwap_dist:.1f}% below VWAP")

    # ------ BEAR CALL SCORING ------
    bear, bear_r = 0, []
    if rsi > 70: bear += 25; bear_r.append(f"RSI {rsi:.1f} overbought")
    elif rsi > 60: bear += 12; bear_r.append(f"RSI {rsi:.1f} elevated")
    if -3 < dist_r < 1: bear += 20; bear_r.append(f"Near resistance ({dist_r:+.1f}%)")
    if not np.isnan(hv_rank) and hv_rank > 40: bear += 15; bear_r.append(f"HV Rank {hv_rank:.0f}")
    if not above_200: bear += 12; bear_r.append("Below 200-SMA")
    if up >= 2: bear += min(12, 4 * up); bear_r.append(f"{up}d up streak")

    # MFI overbought
    if not np.isnan(mfi):
        if mfi > 80: bear += 15; bear_r.append(f"MFI {mfi:.0f} overbought (distribution)")
        elif mfi > 70: bear += 8; bear_r.append(f"MFI {mfi:.0f} elevated")

    # Z-score: statistically overbought
    if not np.isnan(zscore):
        if zscore > 2: bear += 15; bear_r.append(f"Z-score {zscore:.1f} (extreme overbought)")
        elif zscore > 1: bear += 8; bear_r.append(f"Z-score {zscore:.1f} (overbought)")

    # ADX
    if not np.isnan(adx):
        if 20 <= adx <= 40: bear += 8; bear_r.append(f"ADX {adx:.0f} (healthy trend)")
        elif adx > 40: bear -= 5; bear_r.append(f"ADX {adx:.0f} (overextended trend)")
        elif adx < 15: bear -= 3; bear_r.append(f"ADX {adx:.0f} (choppy/no trend)")

    # Volume ratio on a rally — heavy volume = real distribution at the top
    if not np.isnan(vol_ratio) and up >= 1:
        if vol_ratio > 1.5: bear += 10; bear_r.append(f"Rally on heavy volume ({vol_ratio:.1f}x — climax?)")
        elif vol_ratio < 0.8: bear += 5; bear_r.append(f"Rally on weak volume ({vol_ratio:.1f}x — no conviction)")

    # VWAP distance — well above VWAP = stretched
    if not np.isnan(vwap_dist):
        if vwap_dist > 3: bear += 8; bear_r.append(f"{vwap_dist:.1f}% above VWAP")
        elif vwap_dist > 1.5: bear += 4; bear_r.append(f"{vwap_dist:.1f}% above VWAP")

    if bull >= 60 and bull > bear: return "BULL PUT", bull, bull_r
    if bear >= 50 and bear > bull: return "BEAR CALL", bear, bear_r
    if bull >= 35: return "WATCH (bull)", bull, bull_r
    if bear >= 30: return "WATCH (bear)", bear, bear_r
    return "NEUTRAL", max(bull, bear), ["No clear setup"]


def build_spread(row: dict, signal: str, dte: int) -> dict | None:
    price, sigma = row["Price"], row["Sigma"]
    T = dte / 365.0
    width = strike_width(price)

    if "BULL" in signal or signal == "WATCH (bull)":
        target = price * 0.95
        if row["Support"] < price and row["Support"] > price * 0.92:
            target = max(target, row["Support"])
        short = round_strike(target)
        long_ = round_strike(short - width)
        s_val = bs_price(price, short, T, sigma, False)
        l_val = bs_price(price, long_, T, sigma, False)
        credit = s_val - l_val
        spread_w = short - long_
        max_loss = spread_w - credit
        be = short - credit
        pop = prob_above(price, be, T, sigma) * 100
        roc = (credit / max_loss * 100) if max_loss > 0 else 0
        return {"type": "BULL PUT SPREAD", "short": short, "long": long_,
                "credit": credit, "max_profit": credit, "max_loss": max_loss,
                "breakeven": be, "pop": pop, "roc": roc}

    if "BEAR" in signal or signal == "WATCH (bear)":
        target = price * 1.05
        if row["Resistance"] > price and row["Resistance"] < price * 1.08:
            target = min(target, row["Resistance"])
        short = round_strike(target)
        long_ = round_strike(short + width)
        s_val = bs_price(price, short, T, sigma, True)
        l_val = bs_price(price, long_, T, sigma, True)
        credit = s_val - l_val
        spread_w = long_ - short
        max_loss = spread_w - credit
        be = short + credit
        pop = prob_below(price, be, T, sigma) * 100
        roc = (credit / max_loss * 100) if max_loss > 0 else 0
        return {"type": "BEAR CALL SPREAD", "short": short, "long": long_,
                "credit": credit, "max_profit": credit, "max_loss": max_loss,
                "breakeven": be, "pop": pop, "roc": roc}
    return None


def value_spread(trade: PaperTrade, current_price: float, sigma: float) -> dict:
    exp_date = datetime.strptime(trade.expiration, "%Y-%m-%d")
    dte = max(0, (exp_date - datetime.now()).days)
    T = dte / 365.0
    is_call = "CALL" in trade.spread_type
    short_val = bs_price(current_price, trade.short_strike, T, sigma, is_call)
    long_val = bs_price(current_price, trade.long_strike, T, sigma, is_call)
    current_debit = short_val - long_val
    pnl_per_share = trade.entry_credit - current_debit
    pnl_total = pnl_per_share * 100 * trade.contracts
    pct_of_max = (pnl_per_share / trade.max_profit * 100) if trade.max_profit > 0 else 0
    if is_call:
        be_distance = (trade.breakeven - current_price) / current_price * 100
    else:
        be_distance = (current_price - trade.breakeven) / current_price * 100
    return {"dte": dte, "current_debit": current_debit,
            "pnl_per_share": pnl_per_share, "pnl_total": pnl_total,
            "pct_of_max": pct_of_max, "be_distance_pct": be_distance,
            "current_price": current_price}


# ===========================================================================
# Streamlit UI
# ===========================================================================
st.set_page_config(page_title="Credit Spread Trader", page_icon="📈", layout="wide")

st.title("📈 Credit Spread Scanner & Paper Trader")
st.caption("Educational tool — not financial advice. Always verify on a live options chain.")

tab_scan, tab_open, tab_track, tab_history, tab_settings = st.tabs(
    ["🔍 Scanner", "➕ Open Trade", "📊 Open Trades", "📋 History", "⚙️ Settings"]
)

# ---------------------------------------------------------------------------
# TAB 1: Scanner
# ---------------------------------------------------------------------------
with tab_scan:
    st.subheader("Scan stocks for credit spread setups")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        ticker_input = st.text_area(
            "Tickers (comma-separated, leave blank to use default universe)",
            value="", height=70,
            placeholder="e.g. BAC, AAPL, JPM, GOOGL"
        )
    with col2:
        target_dte = st.number_input(
            "Target DTE", min_value=30, max_value=90, value=30, step=1,
            help="Minimum 30 DTE — shorter expirations have too much gamma risk for credit spreads",
        )
    with col3:
        st.write("")
        st.write("")
        run_scan = st.button("Run Scan", type="primary", use_container_width=True)

    # Auto-track controls
    at1, at2, at3 = st.columns([1, 1, 2])
    with at1:
        auto_track_enabled = st.checkbox(
            "🤖 Auto-track high scores",
            value=True,
            help="Automatically open a paper trade for every setup with score ≥ threshold",
        )
    with at2:
        auto_track_threshold = st.number_input(
            "Threshold", min_value=50, max_value=200, value=75, step=5,
            disabled=not auto_track_enabled,
        )
    with at3:
        st.caption("Auto-tracked trades are tagged 🤖 in your history so you can compare "
                   "the scanner's edge vs your manual picks.")

    if run_scan:
        tickers = (
            [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            if ticker_input.strip() else DEFAULT_TICKERS
        )
        exp = next_monthly_expiration(target_dte=target_dte)
        dte = (exp - datetime.now()).days

        progress = st.progress(0.0, text="Scanning...")
        rows = []
        for i, t in enumerate(tickers):
            progress.progress((i + 1) / len(tickers), text=f"Analyzing {t}...")
            data = analyze_ticker(t)
            if data is None:
                continue
            signal, score, reasons = classify(data)
            data["Signal"] = signal
            data["Score"] = score
            data["Reasons"] = "; ".join(reasons)
            spread = build_spread(data, signal, dte) if signal != "NEUTRAL" and signal != "AVOID" else None
            if spread:
                data["Spread_Type"] = spread["type"]
                data["Short_Strike"] = spread["short"]
                data["Long_Strike"] = spread["long"]
                data["Net_Credit"] = round(spread["credit"], 2)
                data["Max_Profit"] = round(spread["max_profit"] * 100, 2)
                data["Max_Loss"] = round(spread["max_loss"] * 100, 2)
                data["Breakeven"] = round(spread["breakeven"], 2)
                data["POP_%"] = round(spread["pop"], 1)
                data["ROC_%"] = round(spread["roc"], 1)
                data["Expiration"] = exp.strftime("%Y-%m-%d")
                data["DTE"] = dte
            rows.append(data)
        progress.empty()

        if not rows:
            st.warning("No data returned for any of the requested tickers.")
        else:
            df = pd.DataFrame(rows).sort_values(["Score"], ascending=False).reset_index(drop=True)
            actionable = df[df["Signal"].isin(["BULL PUT", "BEAR CALL"])]
            watch = df[df["Signal"].astype(str).str.startswith("WATCH")]

            # ---- AUTO-TRACK HIGH SCORES ----
            if auto_track_enabled and not actionable.empty:
                high_scores = actionable[actionable["Score"] >= auto_track_threshold]
                if not high_scores.empty:
                    existing_trades = load_trades()
                    # Dedupe key: ticker + strikes + expiration. If the same setup is
                    # already tracked (open or recently closed), don't add another.
                    today_iso = datetime.now().strftime("%Y-%m-%d")

                    def is_duplicate(row, existing) -> bool:
                        for et in existing:
                            if (
                                et.ticker == row["Ticker"]
                                and et.spread_type == row["Spread_Type"]
                                and abs(et.short_strike - row["Short_Strike"]) < 0.01
                                and abs(et.long_strike - row["Long_Strike"]) < 0.01
                                and et.expiration == row["Expiration"]
                                # Ignore trades closed > 7 days ago — same signal could legitimately retrigger
                                and (et.status == "OPEN"
                                     or (et.close_date >= (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")))
                            ):
                                return True
                        return False

                    added = []
                    for _, hr in high_scores.iterrows():
                        if is_duplicate(hr, existing_trades):
                            continue
                        # Build the auto-tracked paper trade
                        is_call = "CALL" in hr["Spread_Type"]
                        entry_credit = float(hr["Net_Credit"])
                        width = abs(hr["Long_Strike"] - hr["Short_Strike"])
                        max_loss_per_share = width - entry_credit
                        breakeven = (
                            hr["Short_Strike"] + entry_credit if is_call
                            else hr["Short_Strike"] - entry_credit
                        )
                        new_trade = PaperTrade(
                            trade_id=str(uuid.uuid4())[:8],
                            ticker=hr["Ticker"],
                            spread_type=hr["Spread_Type"],
                            short_strike=float(hr["Short_Strike"]),
                            long_strike=float(hr["Long_Strike"]),
                            expiration=hr["Expiration"],
                            contracts=1,
                            entry_date=today_iso,
                            entry_price=float(hr["Price"]),
                            entry_credit=entry_credit,
                            max_profit=entry_credit,
                            max_loss=max_loss_per_share,
                            breakeven=breakeven,
                            notes=f"AUTO: {hr['Reasons']}",
                            source="AUTO",
                            scanner_score=int(hr["Score"]),
                        )
                        existing_trades.append(new_trade)
                        added.append(new_trade)

                    if added:
                        save_trades(existing_trades)
                        st.success(
                            f"🤖 Auto-tracked {len(added)} high-score setup(s) "
                            f"(score ≥ {auto_track_threshold}): "
                            + ", ".join(t.ticker for t in added)
                        )
                    elif len(high_scores) > 0:
                        st.info(
                            f"🤖 Found {len(high_scores)} high-score setup(s) but they were "
                            f"already auto-tracked from a previous scan."
                        )

            st.metric("Actionable setups", len(actionable))

            if not actionable.empty:
                st.markdown("### 🎯 Actionable Setups")
                show_cols = ["Ticker", "Signal", "Score", "Price", "RSI14", "MFI",
                             "ZScore", "ADX", "VWAP_Dist_%", "Volume_Ratio", "HV_Rank",
                             "Spread_Type", "Short_Strike", "Long_Strike",
                             "Net_Credit", "Max_Loss", "POP_%", "ROC_%"]
                avail_cols = [c for c in show_cols if c in actionable.columns]
                st.dataframe(
                    actionable[avail_cols].round(2),
                    use_container_width=True, hide_index=True,
                )

                # Detail expander per actionable setup
                st.markdown("### Trade Details")
                for _, r in actionable.iterrows():
                    with st.expander(
                        f"{r['Ticker']} — {r['Spread_Type']} (Score: {r['Score']})"
                    ):
                        # Top row: price + spread metrics
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Price", f"${r['Price']:.2f}")
                        c1.metric("RSI(14)", f"{r['RSI14']:.1f}")
                        c2.metric("Net Credit", f"${r['Net_Credit']:.2f}/sh")
                        c2.metric("Max Loss", f"${r['Max_Loss']:.0f}")
                        c3.metric("POP", f"{r['POP_%']:.0f}%")
                        c3.metric("ROC", f"{r['ROC_%']:.1f}%")

                        # Indicator panel — only show if values are present
                        st.markdown("**Indicators**")
                        ic1, ic2, ic3, ic4, ic5 = st.columns(5)

                        mfi_val = r.get("MFI", float("nan"))
                        if not pd.isna(mfi_val):
                            mfi_label = "oversold" if mfi_val < 20 else ("overbought" if mfi_val > 80 else "neutral")
                            ic1.metric("MFI", f"{mfi_val:.0f}", help=f"Money Flow Index — {mfi_label}")

                        z_val = r.get("ZScore", float("nan"))
                        if not pd.isna(z_val):
                            z_label = "extreme oversold" if z_val < -2 else ("extreme overbought" if z_val > 2 else "normal")
                            ic2.metric("Z-Score", f"{z_val:+.1f}", help=f"20-day Z-score — {z_label}")

                        adx_val = r.get("ADX", float("nan"))
                        if not pd.isna(adx_val):
                            adx_label = "no trend" if adx_val < 20 else ("strong" if adx_val > 40 else "trending")
                            ic3.metric("ADX", f"{adx_val:.0f}", help=f"Trend strength — {adx_label}")

                        vwap_val = r.get("VWAP_Dist_%", float("nan"))
                        if not pd.isna(vwap_val):
                            ic4.metric("VWAP Dist", f"{vwap_val:+.1f}%",
                                       help="Percent above (or below) 20-day VWAP")

                        vr_val = r.get("Volume_Ratio", float("nan"))
                        if not pd.isna(vr_val):
                            vr_label = "heavy" if vr_val > 1.5 else ("light" if vr_val < 0.8 else "normal")
                            ic5.metric("Vol vs Avg", f"{vr_val:.1f}x", help=f"Today vs 20-day avg — {vr_label}")

                        st.markdown(f"**Trade:** Sell {r['Spread_Type'].split()[1].lower()} @ ${r['Short_Strike']:.2f} / "
                                    f"Buy @ ${r['Long_Strike']:.2f}")
                        st.markdown(f"**Expiration:** {r['Expiration']} ({r['DTE']} DTE)")
                        st.markdown(f"**Breakeven:** ${r['Breakeven']:.2f}")
                        st.markdown(f"**Reasons:** {r['Reasons']}")

                        # Streak flag
                        down, up = int(r["Down_Streak"]), int(r["Up_Streak"])
                        if down >= 4: st.warning(f"🚨 {down} consecutive DOWN closes (capitulation watch)")
                        elif down == 3: st.info(f"📉 3 consecutive DOWN closes (strong pullback)")
                        elif down == 2: st.info(f"📉 2 consecutive DOWN closes (early pullback)")
                        elif up >= 4: st.warning(f"🚀 {up} consecutive UP closes (extended rally)")
                        elif up == 3: st.info(f"📈 3 consecutive UP closes (strong rally)")
                        elif up == 2: st.info(f"📈 2 consecutive UP closes (early rally)")

                        # One-click paper trade button — prefills the Open Trade tab
                        st.markdown("---")
                        if st.button(
                            f"📝 Paper Trade This Setup",
                            key=f"papertrade_{r['Ticker']}",
                            use_container_width=True,
                        ):
                            st.session_state["prefill_trade"] = {
                                "ticker": r["Ticker"],
                                "spread_type": r["Spread_Type"],
                                "short_strike": float(r["Short_Strike"]),
                                "long_strike": float(r["Long_Strike"]),
                                "expiration": r["Expiration"],
                                "estimated_credit": float(r["Net_Credit"]),
                                "notes": f"From scanner: {r['Reasons']}",
                            }
                            # Bump the key counter so Open Trade widgets get fresh state
                            st.session_state["form_key_counter"] = (
                                st.session_state.get("form_key_counter", 0) + 1
                            )
                            st.success("✅ Setup loaded! Switching to Open Trade tab...")
                            st.rerun()

                # CSV download
                csv = actionable.to_csv(index=False).encode()
                st.download_button("Download actionable setups (CSV)", csv,
                                   file_name=f"spreads_{datetime.now().strftime('%Y%m%d')}.csv")
            else:
                st.info("No actionable setups — try a different ticker list or check back tomorrow.")

            if not watch.empty:
                with st.expander(f"Watch list ({len(watch)} tickers)"):
                    show_cols = ["Ticker", "Signal", "Score", "Price", "RSI14", "HV_Rank", "Reasons"]
                    avail_cols = [c for c in show_cols if c in watch.columns]
                    st.dataframe(watch[avail_cols].round(2), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2: Open Trade
# ---------------------------------------------------------------------------
with tab_open:
    st.subheader("Open a new paper trade")

    # Pull prefill data if user clicked "Paper Trade This" in scanner
    prefill = st.session_state.get("prefill_trade", {})
    # Counter-based suffix forces widget recreation when prefill changes
    fk = st.session_state.get("form_key_counter", 0)

    if prefill:
        st.success(f"📋 Pre-filled from scanner: **{prefill['ticker']}** {prefill['spread_type']}  "
                   f"@ {prefill['short_strike']:.0f}/{prefill['long_strike']:.0f}, "
                   f"exp {prefill['expiration']}, est. credit ${prefill['estimated_credit']:.2f}")
        if st.button("Clear prefill"):
            st.session_state.pop("prefill_trade", None)
            st.session_state["form_key_counter"] = fk + 1
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input(
            "Ticker", value=prefill.get("ticker", ""),
            key=f"open_ticker_{fk}",
        ).upper().strip()

        spread_type = st.selectbox(
            "Spread type", ["BULL PUT SPREAD", "BEAR CALL SPREAD"],
            index=1 if prefill.get("spread_type") == "BEAR CALL SPREAD" else 0,
            key=f"open_spread_type_{fk}",
        )
        is_call = "CALL" in spread_type

        short_strike = st.number_input(
            "Short strike (sell)", min_value=0.0, step=0.5,
            value=float(prefill.get("short_strike", 0.0)),
            key=f"open_short_{fk}",
        )

        # Smart default for long strike based on short strike + spread type
        suggested_long = 0.0
        if short_strike > 0:
            default_width = strike_width(short_strike)
            if is_call:
                suggested_long = short_strike + default_width
            else:
                suggested_long = max(0.0, short_strike - default_width)

        long_strike = st.number_input(
            "Long strike (buy)", min_value=0.0, step=0.5,
            value=float(prefill.get("long_strike", suggested_long)),
            key=f"open_long_{fk}",
            help=f"Suggested width: ${strike_width(short_strike) if short_strike > 0 else 0:.2f}",
        )

    with col2:
        # Default to at least 30 days out, snapped to the next monthly expiration
        if prefill.get("expiration"):
            default_exp = datetime.strptime(prefill["expiration"], "%Y-%m-%d").date()
        else:
            default_exp = next_monthly_expiration(target_dte=30).date()

        expiration = st.date_input(
            "Expiration", value=default_exp, key=f"open_exp_{fk}",
            min_value=(datetime.now() + timedelta(days=30)).date(),
            help="Must be at least 30 days out — credit spreads under 30 DTE carry too much gamma risk",
        )

        contracts = st.number_input(
            "Contracts", min_value=1, max_value=100, value=1, step=1,
            key=f"open_contracts_{fk}",
        )

        entry_mode = st.radio(
            "Entry pricing",
            ["Black-Scholes estimate", "Net credit (manual)", "Leg fills (manual)"],
            key=f"open_entry_mode_{fk}",
        )

    valid_setup = (
        ticker
        and short_strike > 0
        and long_strike > 0
        and (expiration - datetime.now().date()).days >= 30
        and (
            (spread_type == "BULL PUT SPREAD" and long_strike < short_strike)
            or (spread_type == "BEAR CALL SPREAD" and long_strike > short_strike)
        )
    )

    manual_credit = 0.0
    short_fill = 0.0
    long_fill = 0.0
    if entry_mode == "Net credit (manual)":
        manual_credit = st.number_input(
            "Net credit per share ($)", min_value=0.0, step=0.01,
            value=float(prefill.get("estimated_credit", 0.0)),
            key=f"open_credit_{fk}",
        )
    elif entry_mode == "Leg fills (manual)":
        cf1, cf2 = st.columns(2)
        with cf1:
            short_fill = st.number_input(
                f"SOLD {('call' if is_call else 'put')} fill ($/share)",
                min_value=0.0, step=0.01, value=0.0, key=f"open_short_fill_{fk}",
            )
        with cf2:
            long_fill = st.number_input(
                f"BOUGHT {('call' if is_call else 'put')} fill ($/share)",
                min_value=0.0, step=0.01, value=0.0, key=f"open_long_fill_{fk}",
            )

    notes = st.text_input(
        "Notes (optional)", value=prefill.get("notes", ""), key=f"open_notes_{fk}",
    )

    # ---- LIVE PREVIEW PANEL ----
    if valid_setup:
        with st.spinner(f"Pricing {ticker}..."):
            res = get_price_and_vol(ticker)

        if res is not None:
            price, sigma = res
            dte = (expiration - datetime.now().date()).days
            T = max(dte / 365.0, 1e-6)

            # Compute estimated credit based on mode
            if entry_mode == "Net credit (manual)":
                preview_credit = manual_credit
            elif entry_mode == "Leg fills (manual)":
                preview_credit = short_fill - long_fill
            else:
                s_val = bs_price(price, short_strike, T, sigma, is_call)
                l_val = bs_price(price, long_strike, T, sigma, is_call)
                preview_credit = s_val - l_val

            width = abs(long_strike - short_strike)
            preview_max_loss = width - preview_credit
            preview_breakeven = (
                short_strike + preview_credit if is_call else short_strike - preview_credit
            )

            # POP estimate using risk-neutral lognormal
            if is_call:
                pop = prob_below(price, preview_breakeven, T, sigma) * 100
            else:
                pop = prob_above(price, preview_breakeven, T, sigma) * 100
            roc = (preview_credit / preview_max_loss * 100) if preview_max_loss > 0 else 0

            st.markdown("---")
            st.markdown("### 📊 Live Trade Preview")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Underlying", f"${price:.2f}")
            p2.metric("DTE", f"{dte} days")
            p3.metric("Total credit", f"${preview_credit * 100 * contracts:.2f}")
            p4.metric("Capital at risk", f"${preview_max_loss * 100 * contracts:.2f}")

            p5, p6, p7, p8 = st.columns(4)
            p5.metric("Net credit", f"${preview_credit:.2f}/sh")
            p6.metric("Breakeven", f"${preview_breakeven:.2f}")
            p7.metric("Prob. of profit", f"{pop:.0f}%")
            p8.metric("Return on risk", f"{roc:.1f}%")

            # BS estimate cross-check when manual mode
            if entry_mode != "Black-Scholes estimate":
                s_val = bs_price(price, short_strike, T, sigma, is_call)
                l_val = bs_price(price, long_strike, T, sigma, is_call)
                bs_credit = s_val - l_val
                if abs(bs_credit - preview_credit) > 0.10:
                    diff = ((preview_credit - bs_credit) / max(bs_credit, 0.01)) * 100
                    st.caption(f"ℹ️ BS estimate is ${bs_credit:.2f}/sh — your entry is "
                               f"{diff:+.0f}% vs theoretical mid")

            # Quick warnings
            if preview_credit <= 0:
                st.error("⚠️ Net credit is zero or negative — this is a debit spread, not a credit spread.")
            if preview_max_loss <= 0:
                st.error("⚠️ Max loss is non-positive — check your strikes.")
            if pop < 50:
                st.warning(f"Low probability of profit ({pop:.0f}%). Consider further OTM strikes.")
            if roc > 100:
                st.warning(f"Very high ROC ({roc:.0f}%) — low probability of profit usually drives this. Verify.")
        else:
            st.warning(f"Could not fetch market data for {ticker}.")
    else:
        # Distinguish "incomplete" from "expiration too close"
        if (
            ticker and short_strike > 0 and long_strike > 0
            and (expiration - datetime.now().date()).days < 30
        ):
            dte_now = (expiration - datetime.now().date()).days
            st.warning(f"⚠️ Expiration is only {dte_now} DTE — minimum is 30 days. Pick a later date.")
        else:
            st.info("Fill in ticker, strikes, and an expiration ≥ 30 days out to see live preview.")

    # ---- SUBMIT ----
    st.markdown("---")
    submit = st.button("✅ Record paper trade", type="primary", use_container_width=True)

    if submit:
        errors = []
        if not ticker:
            errors.append("Ticker required.")
        if short_strike <= 0 or long_strike <= 0:
            errors.append("Both strikes must be positive.")
        if spread_type == "BULL PUT SPREAD" and long_strike >= short_strike:
            errors.append("Bull put: long strike must be BELOW short strike.")
        if spread_type == "BEAR CALL SPREAD" and long_strike <= short_strike:
            errors.append("Bear call: long strike must be ABOVE short strike.")
        dte_check = (expiration - datetime.now().date()).days
        if dte_check < 30:
            errors.append(f"Expiration must be at least 30 days out (currently {dte_check} DTE).")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner(f"Fetching {ticker}..."):
                res = get_price_and_vol(ticker)
            if res is None:
                st.error(f"Could not fetch market data for {ticker}.")
            else:
                price, sigma = res
                exp_str = expiration.strftime("%Y-%m-%d")
                dte = (expiration - datetime.now().date()).days
                T = dte / 365.0

                if entry_mode == "Net credit (manual)":
                    entry_credit = manual_credit
                elif entry_mode == "Leg fills (manual)":
                    entry_credit = short_fill - long_fill
                else:
                    s_val = bs_price(price, short_strike, T, sigma, is_call)
                    l_val = bs_price(price, long_strike, T, sigma, is_call)
                    entry_credit = s_val - l_val

                width = abs(long_strike - short_strike)
                breakeven = short_strike + entry_credit if is_call else short_strike - entry_credit

                trade = PaperTrade(
                    trade_id=str(uuid.uuid4())[:8],
                    ticker=ticker, spread_type=spread_type,
                    short_strike=short_strike, long_strike=long_strike,
                    expiration=exp_str, contracts=int(contracts),
                    entry_date=datetime.now().strftime("%Y-%m-%d"),
                    entry_price=price, entry_credit=entry_credit,
                    max_profit=entry_credit, max_loss=width - entry_credit,
                    breakeven=breakeven, notes=notes,
                )
                trades = load_trades()
                trades.append(trade)
                save_trades(trades)

                # Clear prefill so next entry is fresh
                st.session_state.pop("prefill_trade", None)

                st.success(f"✅ Trade recorded ({trade.trade_id}) — see Open Trades tab to track it")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total credit", f"${entry_credit * 100 * contracts:.2f}")
                c2.metric("Max profit", f"${entry_credit * 100 * contracts:.2f}")
                c3.metric("Max loss", f"${(width - entry_credit) * 100 * contracts:.2f}")


# ---------------------------------------------------------------------------
# TAB 3: Open Trades
# ---------------------------------------------------------------------------
with tab_track:
    st.subheader("Open paper trades")

    trades = load_trades()
    open_trades = [t for t in trades if t.status == "OPEN"]

    # Top action bar
    bar1, bar2, bar3 = st.columns([1, 1, 2])
    with bar1:
        if st.button("🔄 Refresh prices", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with bar2:
        # Bulk-close expired trades
        expired_ids = [
            t.trade_id for t in open_trades
            if (datetime.strptime(t.expiration, "%Y-%m-%d").date() <= datetime.now().date())
        ]
        if st.button(
            f"⏰ Close {len(expired_ids)} expired",
            disabled=(len(expired_ids) == 0),
            use_container_width=True,
            help="Close all expired trades as 'expired worthless' — assumes they finished OTM",
        ):
            for t in trades:
                if t.trade_id in expired_ids:
                    t.status = "CLOSED"
                    t.close_date = datetime.now().strftime("%Y-%m-%d")
                    t.close_debit = 0.0
                    t.realized_pnl = t.entry_credit * 100 * t.contracts
                    t.close_reason = "expired"
            save_trades(trades)
            st.success(f"Closed {len(expired_ids)} expired trades as max-profit.")
            st.rerun()

    if not open_trades:
        st.info("No open trades. Go to **Open Trade** tab to record one, or run the scanner.")
    else:
        # Build summary table with current valuations
        rows = []
        valuations = {}
        for t in open_trades:
            res = get_price_and_vol(t.ticker)
            if res is None:
                continue
            price, sigma = res
            v = value_spread(t, price, sigma)
            valuations[t.trade_id] = (t, price, v)

            # Status emoji for at-a-glance read
            if v["pct_of_max"] >= 50:
                status_emoji = "💰"
            elif v["pnl_per_share"] <= -2 * t.entry_credit:
                status_emoji = "🛑"
            elif v["be_distance_pct"] < 0:
                status_emoji = "⚠️"
            elif v["dte"] <= 7:
                status_emoji = "⏰"
            elif v["pnl_total"] > 0:
                status_emoji = "✅"
            else:
                status_emoji = "—"

            rows.append({
                "": status_emoji,
                "Src": "🤖" if t.source == "AUTO" else "👤",
                "ID": t.trade_id, "Ticker": t.ticker,
                "Type": "Bull Put" if "PUT" in t.spread_type else "Bear Call",
                "Strikes": f"{t.short_strike:.0f}/{t.long_strike:.0f}",
                "Score": t.scanner_score if t.source == "AUTO" else "—",
                "Exp": t.expiration, "DTE": v["dte"],
                "Underlying": f"${price:.2f}",
                "Entry $": f"${t.entry_credit:.2f}",
                "Now $": f"${v['current_debit']:.2f}",
                "P&L": f"${v['pnl_total']:+.0f}",
                "% Max": f"{v['pct_of_max']:+.0f}%",
            })

        if rows:
            total_unrealized = sum(v[2]["pnl_total"] for v in valuations.values())
            winners = sum(1 for v in valuations.values() if v[2]["pnl_total"] > 0)
            losers = sum(1 for v in valuations.values() if v[2]["pnl_total"] < 0)

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Total unrealized P&L", f"${total_unrealized:+.2f}")
            sm2.metric("Open positions", len(valuations))
            sm3.metric("Winning", winners)
            sm4.metric("Losing", losers)

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("💰 = at profit target  🛑 = stop-loss territory  ⚠️ = past breakeven  "
                       "⏰ = expiring soon  ✅ = winning  — = neutral")

            # Detail + close form for each open trade
            st.markdown("### Manage trades")
            for tid, (t, price, v) in valuations.items():
                # Status emoji in header for quick scan
                header_emoji = "💰" if v["pct_of_max"] >= 50 else (
                    "🛑" if v["pnl_per_share"] <= -2 * t.entry_credit else (
                        "⚠️" if v["be_distance_pct"] < 0 else (
                            "⏰" if v["dte"] <= 7 else ""
                        )
                    )
                )
                with st.expander(
                    f"{header_emoji} [{t.trade_id}] {t.ticker} {t.spread_type} — "
                    f"P&L ${v['pnl_total']:+.0f} ({v['pct_of_max']:+.0f}% of max)"
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Underlying", f"${price:.2f}", f"{(price - t.entry_price):+.2f}")
                    c2.metric("Current debit", f"${v['current_debit']:.2f}")
                    c3.metric("Days remaining", f"{v['dte']}")

                    st.write(f"**Entry:** ${t.entry_credit:.2f}/sh on {t.entry_date} when price was ${t.entry_price:.2f}")
                    st.write(f"**Breakeven:** ${t.breakeven:.2f} ({v['be_distance_pct']:+.1f}% cushion)")
                    if t.notes:
                        st.write(f"**Notes:** {t.notes}")

                    # Action hints
                    if v["pct_of_max"] >= 50:
                        st.success("💰 At 50%+ of max profit — consider closing (standard profit target)")
                    elif v["pnl_per_share"] <= -2 * t.entry_credit:
                        st.error("🛑 Loss exceeds 2x credit collected — stop-loss territory")
                    elif v["be_distance_pct"] < 0:
                        st.warning("⚠️ Underlying has crossed breakeven against you")
                    elif v["dte"] <= 7:
                        st.info("⏰ Under 1 week to expiration — gamma risk increasing")

                    # ---- QUICK ACTIONS ----
                    st.markdown("**Quick close**")
                    qa1, qa2, qa3, qa4 = st.columns(4)

                    # Quick close at profit target (50% of max)
                    target_debit = t.entry_credit * 0.5
                    target_pnl = (t.entry_credit - target_debit) * 100 * t.contracts
                    with qa1:
                        if st.button(
                            f"💰 50% profit\n${target_pnl:+.0f}",
                            key=f"qc_50_{t.trade_id}",
                            use_container_width=True,
                            help=f"Close at debit ${target_debit:.2f} (50% of credit)",
                        ):
                            t.status = "CLOSED"
                            t.close_date = datetime.now().strftime("%Y-%m-%d")
                            t.close_debit = target_debit
                            t.realized_pnl = target_pnl
                            t.close_reason = "profit_target"
                            save_trades(trades)
                            st.rerun()

                    # Quick close at current BS price
                    with qa2:
                        if st.button(
                            f"📊 Mid price\n${v['pnl_total']:+.0f}",
                            key=f"qc_mid_{t.trade_id}",
                            use_container_width=True,
                            help=f"Close at current BS estimate ${v['current_debit']:.2f}",
                        ):
                            t.status = "CLOSED"
                            t.close_date = datetime.now().strftime("%Y-%m-%d")
                            t.close_debit = v["current_debit"]
                            t.realized_pnl = v["pnl_total"]
                            t.close_reason = "manual"
                            save_trades(trades)
                            st.rerun()

                    # Quick close at expired worthless (max profit)
                    max_profit = t.entry_credit * 100 * t.contracts
                    with qa3:
                        if st.button(
                            f"⏰ Expired\n${max_profit:+.0f}",
                            key=f"qc_exp_{t.trade_id}",
                            use_container_width=True,
                            help="Closed at $0 = max profit (only use if expired OTM)",
                        ):
                            t.status = "CLOSED"
                            t.close_date = datetime.now().strftime("%Y-%m-%d")
                            t.close_debit = 0.0
                            t.realized_pnl = max_profit
                            t.close_reason = "expired"
                            save_trades(trades)
                            st.rerun()

                    # Quick close at stop loss (2x credit)
                    sl_debit = t.entry_credit * 3  # buyback at 3x credit = -2x credit P&L
                    sl_pnl = (t.entry_credit - sl_debit) * 100 * t.contracts
                    with qa4:
                        if st.button(
                            f"🛑 Stop loss\n${sl_pnl:+.0f}",
                            key=f"qc_sl_{t.trade_id}",
                            use_container_width=True,
                            help=f"Close at 2x credit loss (debit ${sl_debit:.2f})",
                        ):
                            t.status = "CLOSED"
                            t.close_date = datetime.now().strftime("%Y-%m-%d")
                            t.close_debit = sl_debit
                            t.realized_pnl = sl_pnl
                            t.close_reason = "stop_loss"
                            save_trades(trades)
                            st.rerun()

                    # ---- MANUAL CLOSE (with custom price) ----
                    st.markdown("---")
                    st.markdown("**Custom close**")
                    close_mode = st.radio(
                        "Close pricing", ["BS estimate", "Net debit (manual)", "Leg fills", "Expired worthless"],
                        key=f"close_mode_{t.trade_id}", horizontal=True,
                    )
                    cd1, cd2 = st.columns(2)
                    with cd1:
                        manual_debit = 0.0
                        sb, ls = 0.0, 0.0
                        if close_mode == "Net debit (manual)":
                            manual_debit = st.number_input(
                                "Net debit ($/share)", min_value=0.0, step=0.01,
                                value=float(round(v["current_debit"], 2)),
                                key=f"debit_{t.trade_id}",
                            )
                        elif close_mode == "Leg fills":
                            sb = st.number_input("Short leg buyback", min_value=0.0, step=0.01, value=0.0,
                                                  key=f"sb_{t.trade_id}")
                            ls = st.number_input("Long leg sale", min_value=0.0, step=0.01, value=0.0,
                                                  key=f"ls_{t.trade_id}")
                        reason = st.selectbox(
                            "Reason", ["profit_target", "stop_loss", "expired", "manual"],
                            key=f"reason_{t.trade_id}",
                        )
                    with cd2:
                        st.write("")
                        if st.button(f"Close trade {t.trade_id}", key=f"close_btn_{t.trade_id}", type="primary"):
                            if close_mode == "Net debit (manual)":
                                debit = manual_debit
                            elif close_mode == "Leg fills":
                                debit = sb - ls
                            elif close_mode == "Expired worthless":
                                debit = 0.0
                            else:
                                debit = v["current_debit"]
                            realized = (t.entry_credit - debit) * 100 * t.contracts
                            t.status = "CLOSED"
                            t.close_date = datetime.now().strftime("%Y-%m-%d")
                            t.close_debit = debit
                            t.realized_pnl = realized
                            t.close_reason = reason
                            save_trades(trades)
                            st.success(f"Closed. Realized P&L: ${realized:+.2f}")
                            st.rerun()


# ---------------------------------------------------------------------------
# TAB 4: History
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Trade history & performance")
    trades = load_trades()
    closed = [t for t in trades if t.status == "CLOSED"]

    if not closed:
        st.info("No closed trades yet.")
    else:
        # Helper to compute stats for any subset of trades
        def compute_stats(subset):
            if not subset:
                return None
            total_pnl = sum(t.realized_pnl for t in subset)
            wins = [t for t in subset if t.realized_pnl > 0]
            losses = [t for t in subset if t.realized_pnl < 0]
            win_rate = len(wins) / len(subset) * 100
            avg_win = float(np.mean([t.realized_pnl for t in wins])) if wins else 0.0
            avg_loss = float(np.mean([t.realized_pnl for t in losses])) if losses else 0.0
            pf = (
                sum(t.realized_pnl for t in wins) / abs(sum(t.realized_pnl for t in losses))
                if losses and wins else None
            )
            return {
                "n": len(subset), "wins": len(wins), "losses": len(losses),
                "total_pnl": total_pnl, "win_rate": win_rate,
                "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": pf,
            }

        # Filter selector
        view_filter = st.radio(
            "View",
            ["All trades", "🤖 Auto-tracked only", "👤 Manual only", "Side-by-side comparison"],
            horizontal=True,
        )

        auto_trades = [t for t in closed if t.source == "AUTO"]
        manual_trades = [t for t in closed if t.source == "MANUAL"]

        if view_filter == "Side-by-side comparison":
            st.markdown("### 🤖 vs 👤 Performance Comparison")
            st.caption("Comparing scanner-auto-tracked trades vs your manually entered trades. "
                       "Use this to evaluate whether the scanner's high-score signals actually have edge.")

            ac1, ac2 = st.columns(2)
            with ac1:
                st.markdown("#### 🤖 Auto-tracked")
                stats = compute_stats(auto_trades)
                if stats is None:
                    st.info("No auto-tracked trades closed yet.")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Trades", stats["n"])
                    m2.metric("Win rate", f"{stats['win_rate']:.1f}%")
                    m1.metric("Total P&L", f"${stats['total_pnl']:+.2f}")
                    m2.metric("Avg win", f"${stats['avg_win']:+.2f}")
                    m1.metric("Avg loss", f"${stats['avg_loss']:+.2f}")
                    if stats["profit_factor"] is not None:
                        m2.metric("Profit factor", f"{stats['profit_factor']:.2f}")

            with ac2:
                st.markdown("#### 👤 Manual")
                stats = compute_stats(manual_trades)
                if stats is None:
                    st.info("No manual trades closed yet.")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Trades", stats["n"])
                    m2.metric("Win rate", f"{stats['win_rate']:.1f}%")
                    m1.metric("Total P&L", f"${stats['total_pnl']:+.2f}")
                    m2.metric("Avg win", f"${stats['avg_win']:+.2f}")
                    m1.metric("Avg loss", f"${stats['avg_loss']:+.2f}")
                    if stats["profit_factor"] is not None:
                        m2.metric("Profit factor", f"{stats['profit_factor']:.2f}")

            # Score-bucket analysis if there are enough auto trades
            if len(auto_trades) >= 5:
                st.markdown("### 🎯 Performance by Scanner Score (auto-tracked)")
                buckets = [(75, 79), (80, 89), (90, 99), (100, 200)]
                bucket_rows = []
                for lo, hi in buckets:
                    sub = [t for t in auto_trades if lo <= t.scanner_score <= hi]
                    if sub:
                        s = compute_stats(sub)
                        bucket_rows.append({
                            "Score Range": f"{lo}-{hi}",
                            "Trades": s["n"],
                            "Win Rate": f"{s['win_rate']:.0f}%",
                            "Total P&L": f"${s['total_pnl']:+.0f}",
                            "Avg Win": f"${s['avg_win']:+.0f}",
                            "Avg Loss": f"${s['avg_loss']:+.0f}",
                        })
                if bucket_rows:
                    st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)
                    st.caption("If higher score buckets have meaningfully better win rates, "
                               "the scanner has predictive power. If they're flat across buckets, "
                               "the score is noise.")
        else:
            # Single-view filtered display
            if view_filter == "🤖 Auto-tracked only":
                view_set = auto_trades
                label = "auto-tracked"
            elif view_filter == "👤 Manual only":
                view_set = manual_trades
                label = "manual"
            else:
                view_set = closed
                label = "all"

            if not view_set:
                st.info(f"No {label} trades closed yet.")
            else:
                rows = [{
                    "Source": "🤖" if t.source == "AUTO" else "👤",
                    "ID": t.trade_id, "Ticker": t.ticker,
                    "Type": "Bull Put" if "PUT" in t.spread_type else "Bear Call",
                    "Strikes": f"{t.short_strike:.0f}/{t.long_strike:.0f}",
                    "Score": t.scanner_score if t.source == "AUTO" else "—",
                    "Opened": t.entry_date, "Closed": t.close_date,
                    "Entry $": f"${t.entry_credit:.2f}",
                    "Close $": f"${t.close_debit:.2f}",
                    "P&L $": f"${t.realized_pnl:+.2f}",
                    "Reason": t.close_reason,
                } for t in view_set]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                stats = compute_stats(view_set)
                st.markdown("### Stats")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total P&L", f"${stats['total_pnl']:+.2f}")
                c2.metric("Win rate", f"{stats['win_rate']:.1f}%")
                c3.metric("Avg win", f"${stats['avg_win']:+.2f}")
                c4.metric("Avg loss", f"${stats['avg_loss']:+.2f}")
                if stats["profit_factor"] is not None:
                    st.metric("Profit factor", f"{stats['profit_factor']:.2f}",
                              help=">1.0 = profitable system")


# ---------------------------------------------------------------------------
# TAB 5: Settings
# ---------------------------------------------------------------------------
with tab_settings:
    st.subheader("Settings")

    trades = load_trades()

    # Storage backend status with clear visual indicator
    if is_github_configured():
        st.success(f"💾 **Storage:** {storage_backend_name()}")
        st.caption("✅ Persistent — survives app restarts and inactivity timeouts on Streamlit Cloud.")
    else:
        st.warning(f"💾 **Storage:** {storage_backend_name()}")
        with st.expander("⚠️ Your trades are EPHEMERAL — click here to set up persistent GitHub storage"):
            st.markdown("""
**Why this matters:** Streamlit Cloud restarts your app after periods of
inactivity (typically 7 days). When that happens, locally stored trades
are wiped. To prevent losing your tracking history, configure GitHub-backed
storage.

**One-time setup (5 minutes):**

1. **Create a GitHub Personal Access Token:**
   - Go to <https://github.com/settings/tokens?type=beta>
   - Click "Generate new token" → "Fine-grained token"
   - Token name: e.g. "spread-tracker"
   - Repository access: pick a repo (can be your app's repo or a separate private one)
   - Permissions → Repository permissions → **Contents: Read and write**
   - Click Generate token, copy it immediately (only shown once!)

2. **Add secrets in Streamlit Cloud:**
   - Open your app dashboard at <https://share.streamlit.io>
   - Click your app → ⋮ menu → "Settings" → "Secrets"
   - Paste:

   ```
   GITHUB_TOKEN = "ghp_yourtokenhere"
   GITHUB_REPO = "yourusername/your-repo-name"
   GITHUB_FILE = "trades.json"
   GITHUB_BRANCH = "main"
   ```

   - Save. The app will auto-restart and start using GitHub for storage.

3. **Verify:** Refresh this page. The Storage line above should now show
   "GitHub (yourusername/your-repo-name/trades.json)" with a green checkmark.

**That's it.** Every save creates a git commit, so you also get automatic
version history of your trades.
            """)

    st.write(f"**Total trades:** {len(trades)} ({sum(1 for t in trades if t.status == 'OPEN')} open, "
             f"{sum(1 for t in trades if t.status == 'CLOSED')} closed)")

    # Manual cache refresh — useful if you've edited trades.json directly on GitHub
    if is_github_configured():
        if st.button("🔄 Reload from GitHub"):
            st.session_state.pop("trades_cache", None)
            st.rerun()

    st.markdown("---")
    st.markdown("### Export")
    if trades:
        export = json.dumps([asdict(t) for t in trades], indent=2)
        st.download_button(
            "Download all trades (JSON)", export,
            file_name=f"paper_trades_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )

    st.markdown("---")
    st.markdown("### Import")
    uploaded = st.file_uploader("Upload trades JSON to restore", type=["json"])
    if uploaded:
        try:
            data = json.load(uploaded)
            new_trades = [PaperTrade(**t) for t in data]
            if st.button(f"Replace existing trades with {len(new_trades)} from file"):
                save_trades(new_trades)
                st.success("Restored. Refresh the page.")
        except Exception as e:
            st.error(f"Could not parse file: {e}")

    st.markdown("---")
    st.markdown("### Danger zone")
    with st.expander("Reset — delete ALL trades"):
        st.warning("This permanently deletes every paper trade. Cannot be undone.")
        confirm_text = st.text_input('Type "DELETE" to confirm', key="reset_confirm")
        if st.button("Delete all trades", type="primary"):
            if confirm_text == "DELETE":
                save_trades([])
                st.success("All trades deleted.")
                st.rerun()
            else:
                st.error('You must type "DELETE" exactly.')
