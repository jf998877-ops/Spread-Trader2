"""
Credit Spread Scanner + Paper Trader — Streamlit Web App
=========================================================
Combines the spread scanner and paper trade tracker into a web interface.

To run locally:
    pip install streamlit yfinance pandas numpy scipy
    streamlit run app.py

To deploy free at <yourapp>.streamlit.app:
    1. Push this file (and requirements.txt) to a public GitHub repo
    2. Go to https://share.streamlit.io
    3. Sign in with GitHub, click "New app", point at the repo
    4. Set main file path to app.py
    5. Done — you get a public URL

requirements.txt should contain:
    streamlit
    yfinance
    pandas
    numpy
    scipy
"""

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
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

# Storage in user's home dir so it persists between app runs locally.
# On Streamlit Cloud this is ephemeral — see notes at end of file.
STORAGE = Path.home() / ".spread_paper_trades.json"


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


def load_trades() -> list[PaperTrade]:
    if not STORAGE.exists():
        return []
    try:
        with open(STORAGE, "r") as f:
            data = json.load(f)
        return [PaperTrade(**t) for t in data]
    except Exception:
        return []


def save_trades(trades: list[PaperTrade]) -> None:
    STORAGE.parent.mkdir(parents=True, exist_ok=True)
    with open(STORAGE, "w") as f:
        json.dump([asdict(t) for t in trades], f, indent=2)


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
    min_dte = max(7, target_dte // 2)
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
    }


def classify(row: dict) -> tuple[str, int, list[str]]:
    if row["Avg_Volume_20d"] < 1_000_000:
        return "AVOID", 0, ["Insufficient liquidity"]

    rsi, above_200 = row["RSI14"], row["Above_200SMA"]
    above_50, hv_rank = row["Above_50SMA"], row["HV_Rank"]
    dist_s, dist_r = row["Dist_Support_%"], row["Dist_Resistance_%"]
    down, up = row["Down_Streak"], row["Up_Streak"]

    bull, bull_r = 0, []
    if above_200: bull += 25; bull_r.append("Above 200-SMA")
    if rsi < 35: bull += 25; bull_r.append(f"RSI {rsi:.1f} oversold")
    elif rsi < 45: bull += 15; bull_r.append(f"RSI {rsi:.1f} pulled back")
    if -2 < dist_s < 5: bull += 20; bull_r.append(f"Near support ({dist_s:+.1f}%)")
    if not np.isnan(hv_rank):
        if hv_rank > 50: bull += 20; bull_r.append(f"HV Rank {hv_rank:.0f}")
        elif hv_rank > 30: bull += 10; bull_r.append(f"HV Rank {hv_rank:.0f}")
    if above_50 and above_200: bull += 10; bull_r.append("Above both MAs")
    if down >= 2: bull += min(15, 5 * down); bull_r.append(f"{down}d down streak")

    bear, bear_r = 0, []
    if rsi > 70: bear += 30; bear_r.append(f"RSI {rsi:.1f} overbought")
    elif rsi > 60: bear += 15; bear_r.append(f"RSI {rsi:.1f} elevated")
    if -3 < dist_r < 1: bear += 25; bear_r.append(f"Near resistance ({dist_r:+.1f}%)")
    if not np.isnan(hv_rank) and hv_rank > 40: bear += 20; bear_r.append(f"HV Rank {hv_rank:.0f}")
    if not above_200: bear += 15; bear_r.append("Below 200-SMA")
    if up >= 2: bear += min(15, 5 * up); bear_r.append(f"{up}d up streak")

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
        target_dte = st.number_input("Target DTE", min_value=7, max_value=90, value=30, step=1)
    with col3:
        st.write("")
        st.write("")
        run_scan = st.button("Run Scan", type="primary", use_container_width=True)

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

            st.metric("Actionable setups", len(actionable))

            if not actionable.empty:
                st.markdown("### 🎯 Actionable Setups")
                show_cols = ["Ticker", "Signal", "Score", "Price", "RSI14", "HV_Rank",
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
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Price", f"${r['Price']:.2f}")
                        c1.metric("RSI(14)", f"{r['RSI14']:.1f}")
                        c2.metric("Net Credit", f"${r['Net_Credit']:.2f}/sh")
                        c2.metric("Max Loss", f"${r['Max_Loss']:.0f}")
                        c3.metric("POP", f"{r['POP_%']:.0f}%")
                        c3.metric("ROC", f"{r['ROC_%']:.1f}%")

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
                            st.success("✅ Setup loaded into Open Trade tab. Click that tab to record it.")

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
    if prefill:
        st.success(f"📋 Pre-filled from scanner: **{prefill['ticker']}** {prefill['spread_type']}")
        if st.button("Clear prefill"):
            st.session_state.pop("prefill_trade", None)
            st.rerun()

    # Inputs OUTSIDE a form so we can show live preview as values change.
    # We use a manual submit button at the bottom instead of st.form.
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input(
            "Ticker", value=prefill.get("ticker", ""),
            key="open_ticker"
        ).upper().strip()

        spread_type = st.selectbox(
            "Spread type", ["BULL PUT SPREAD", "BEAR CALL SPREAD"],
            index=1 if prefill.get("spread_type") == "BEAR CALL SPREAD" else 0,
            key="open_spread_type",
        )
        is_call = "CALL" in spread_type

        short_strike = st.number_input(
            "Short strike (sell)", min_value=0.0, step=0.5,
            value=float(prefill.get("short_strike", 0.0)),
            key="open_short",
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
            key="open_long",
            help=f"Suggested width: ${strike_width(short_strike) if short_strike > 0 else 0:.2f}",
        )

    with col2:
        default_exp = (
            datetime.strptime(prefill["expiration"], "%Y-%m-%d").date()
            if prefill.get("expiration")
            else (datetime.now() + timedelta(days=30)).date()
        )
        expiration = st.date_input("Expiration", value=default_exp, key="open_exp")

        contracts = st.number_input(
            "Contracts", min_value=1, max_value=100, value=1, step=1, key="open_contracts"
        )

        entry_mode = st.radio(
            "Entry pricing",
            ["Black-Scholes estimate", "Net credit (manual)", "Leg fills (manual)"],
            key="open_entry_mode",
        )

    # Live preview — only computes if we have valid inputs
    valid_setup = (
        ticker
        and short_strike > 0
        and long_strike > 0
        and expiration > datetime.now().date()
        and (
            (spread_type == "BULL PUT SPREAD" and long_strike < short_strike)
            or (spread_type == "BEAR CALL SPREAD" and long_strike > short_strike)
        )
    )

    # Pricing inputs based on mode
    manual_credit = 0.0
    short_fill = 0.0
    long_fill = 0.0
    if entry_mode == "Net credit (manual)":
        manual_credit = st.number_input(
            "Net credit per share ($)", min_value=0.0, step=0.01,
            value=float(prefill.get("estimated_credit", 0.0)),
            key="open_credit",
        )
    elif entry_mode == "Leg fills (manual)":
        cf1, cf2 = st.columns(2)
        with cf1:
            short_fill = st.number_input(
                f"SOLD {('call' if is_call else 'put')} fill ($/share)",
                min_value=0.0, step=0.01, value=0.0, key="open_short_fill",
            )
        with cf2:
            long_fill = st.number_input(
                f"BOUGHT {('call' if is_call else 'put')} fill ($/share)",
                min_value=0.0, step=0.01, value=0.0, key="open_long_fill",
            )

    notes = st.text_input(
        "Notes (optional)", value=prefill.get("notes", ""), key="open_notes",
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
        st.info("Fill in ticker, strikes, and expiration to see live preview.")

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
        if expiration <= datetime.now().date():
            errors.append("Expiration must be in the future.")

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
                "ID": t.trade_id, "Ticker": t.ticker,
                "Type": "Bull Put" if "PUT" in t.spread_type else "Bear Call",
                "Strikes": f"{t.short_strike:.0f}/{t.long_strike:.0f}",
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
        rows = [{
            "ID": t.trade_id, "Ticker": t.ticker,
            "Type": "Bull Put" if "PUT" in t.spread_type else "Bear Call",
            "Strikes": f"{t.short_strike:.0f}/{t.long_strike:.0f}",
            "Opened": t.entry_date, "Closed": t.close_date,
            "Entry $": f"${t.entry_credit:.2f}",
            "Close $": f"${t.close_debit:.2f}",
            "P&L $": f"${t.realized_pnl:+.2f}",
            "Reason": t.close_reason,
        } for t in closed]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Stats
        total_pnl = sum(t.realized_pnl for t in closed)
        wins = [t for t in closed if t.realized_pnl > 0]
        losses = [t for t in closed if t.realized_pnl < 0]
        win_rate = len(wins) / len(closed) * 100
        avg_win = np.mean([t.realized_pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.realized_pnl for t in losses]) if losses else 0

        st.markdown("### Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total P&L", f"${total_pnl:+.2f}")
        c2.metric("Win rate", f"{win_rate:.1f}%")
        c3.metric("Avg win", f"${avg_win:+.2f}")
        c4.metric("Avg loss", f"${avg_loss:+.2f}")

        if wins and losses:
            pf = sum(t.realized_pnl for t in wins) / abs(sum(t.realized_pnl for t in losses))
            st.metric("Profit factor", f"{pf:.2f}", help=">1.0 = profitable system")


# ---------------------------------------------------------------------------
# TAB 5: Settings
# ---------------------------------------------------------------------------
with tab_settings:
    st.subheader("Settings")

    trades = load_trades()
    st.write(f"**Storage location:** `{STORAGE}`")
    st.write(f"**Total trades:** {len(trades)} ({sum(1 for t in trades if t.status == 'OPEN')} open, "
             f"{sum(1 for t in trades if t.status == 'CLOSED')} closed)")

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
