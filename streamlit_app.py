"""
AC420 Robo-Advisor  —  Lydia Pelton & Sydney Pitts
=================================================
Run with:   streamlit run app.py

HOW THE FILE IS ORGANIZED (read top to bottom)
  1. Assumptions ............ every fixed number the model uses, with its source
  2. Stock universe ......... the stocks the advisor is allowed to pick from
  3. Data ................... prices (Yahoo Finance), Fama-French factors (Ken French library)
  4. Models ................. CAPM + Fama-French regressions, alpha, expected returns, risk
  5. Portfolio math ......... beta, loadings, return, volatility, Sharpe (stocks + T-bills)
  6. "Start" logic .......... age -> target beta -> pick stocks -> Capital Allocation Line
  7. "Improve" logic ........ analyze holdings -> recommended goals -> reweight + substitutes
  8. Display helpers ........ gauge, charts, plain-English explanations
  9. Home page .............. what the app does, key terms, methods & assumptions
 10. "Start from scratch" page
 11. "Improve my portfolio" page
 12. App layout ............. the three tabs
"""

import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize, linprog


# =============================================================================
# 1. ASSUMPTIONS  (change these here; the whole app uses them)
# =============================================================================

# Fama-French factor file (monthly Mkt-RF, SMB, HML, RF). Same data WRDS distributes.
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_CSV.zip")
# Backup: if the download fails, put the CSV from the Ken French site in the app folder
FF_LOCAL_FILE = "F-F_Research_Data_Factors.csv"

# Factor premiums = long-run average of each factor since July 1963,
# the start date Fama & French (1993) use in their original paper.
PREMIUM_START = "1963-07"

# Blume (1971) adjustment: forecast beta = 0.67 x historical beta + 0.33 x 1.0
# (betas tend to drift toward 1 over time). Used for the CAPM beta.
BLUME_WEIGHT = 0.67

# History window for the regressions. 5 years of monthly data (60 points)
# is the most common choice for beta estimates.
DEFAULT_WINDOW_YEARS = 5
WINDOW_OPTIONS = [3, 5, 10]

# A goal counts as "reached" when the portfolio is within these distances of it
TOLERANCE = {"beta": 0.05, "smb": 0.10, "hml": 0.10, "sharpe": 0.01}

# Importance of each goal in the Tab 2 optimizer (bigger = matters more).
# Turnover keeps the recommended weights as close as possible to what you own now.
W_BETA, W_TILT, W_SHARPE, W_TURN = 50.0, 20.0, 50.0, 1.0

# Rule-of-thumb bands for an annual expected Sharpe ratio (our assumption, used for the gauge)
SHARPE_BANDS = [(0.0, 0.3, "Low"), (0.3, 0.5, "Fair"), (0.5, 0.8, "Good"), (0.8, 99, "Very good")]

# T-bills: beta = 0 and they earn the risk-free rate (Lecture 12). In the app they are one
# line called "T-bills (cash)"; in real life you'd buy a T-bill ETF such as SGOV or BIL.
CASH = "T-BILLS"
CASH_NAME = "T-bills (cash)"
CASH_ALIASES = {"CASH", "T-BILLS", "TBILLS", "T-BILL", "TBILL"}
MAX_CASH = 0.75          # at most 75% in T-bills: this is a stock advisor
MIN_CASH = 0.02          # don't bother with T-bills below 2% of the portfolio

# Chart colors: each stock keeps the same color in every chart; T-bills are gray
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B", "#B279A2",
           "#FF9DA6", "#9D755D", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
           "#8C564B", "#E377C2", "#BCBD22", "#17BECF", "#AEC7E8", "#FFBB78", "#98DF8A",
           "#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2", "#DBDB8D", "#9EDAE5", "#393B79",
           "#637939", "#8C6D31"]
CASH_COLOR = "#9E9E9E"
MARKET_COLOR = "#F08C00"   # the "market" marker on the Sharpe gauge (visible in light and dark mode)


# =============================================================================
# 2. STOCK UNIVERSE  ticker: (company, sector, size group)
# =============================================================================
UNIVERSE = {
    # ---- Large companies
    "DUK": ("Duke Energy", "Utilities", "Large"), "SO": ("Southern Co", "Utilities", "Large"),
    "NEE": ("NextEra Energy", "Utilities", "Large"), "AEP": ("American Electric Power", "Utilities", "Large"),
    "ED": ("Consolidated Edison", "Utilities", "Large"), "XEL": ("Xcel Energy", "Utilities", "Large"),
    "KO": ("Coca-Cola", "Consumer Staples", "Large"), "PEP": ("PepsiCo", "Consumer Staples", "Large"),
    "PG": ("Procter & Gamble", "Consumer Staples", "Large"), "CL": ("Colgate-Palmolive", "Consumer Staples", "Large"),
    "GIS": ("General Mills", "Consumer Staples", "Large"), "KMB": ("Kimberly-Clark", "Consumer Staples", "Large"),
    "WMT": ("Walmart", "Consumer Staples", "Large"), "COST": ("Costco", "Consumer Staples", "Large"),
    "MCD": ("McDonald's", "Consumer Discretionary", "Large"), "VZ": ("Verizon", "Communication", "Large"),
    "T": ("AT&T", "Communication", "Large"),
    "JNJ": ("Johnson & Johnson", "Health Care", "Large"), "MRK": ("Merck", "Health Care", "Large"),
    "PFE": ("Pfizer", "Health Care", "Large"), "ABBV": ("AbbVie", "Health Care", "Large"),
    "UNH": ("UnitedHealth", "Health Care", "Large"), "LLY": ("Eli Lilly", "Health Care", "Large"),
    "TMO": ("Thermo Fisher", "Health Care", "Large"), "ABT": ("Abbott Labs", "Health Care", "Large"),
    "JPM": ("JPMorgan Chase", "Financials", "Large"), "BAC": ("Bank of America", "Financials", "Large"),
    "GS": ("Goldman Sachs", "Financials", "Large"), "MS": ("Morgan Stanley", "Financials", "Large"),
    "V": ("Visa", "Financials", "Large"), "MA": ("Mastercard", "Financials", "Large"),
    "BRK-B": ("Berkshire Hathaway", "Financials", "Large"), "AXP": ("American Express", "Financials", "Large"),
    "CAT": ("Caterpillar", "Industrials", "Large"), "DE": ("Deere", "Industrials", "Large"),
    "HON": ("Honeywell", "Industrials", "Large"), "UPS": ("UPS", "Industrials", "Large"),
    "GE": ("GE Aerospace", "Industrials", "Large"), "LMT": ("Lockheed Martin", "Industrials", "Large"),
    "XOM": ("ExxonMobil", "Energy", "Large"), "CVX": ("Chevron", "Energy", "Large"),
    "LIN": ("Linde", "Materials", "Large"),
    "HD": ("Home Depot", "Consumer Discretionary", "Large"), "NKE": ("Nike", "Consumer Discretionary", "Large"),
    "SBUX": ("Starbucks", "Consumer Discretionary", "Large"), "AMZN": ("Amazon", "Consumer Discretionary", "Large"),
    "TSLA": ("Tesla", "Consumer Discretionary", "Large"),
    "AAPL": ("Apple", "Technology", "Large"), "MSFT": ("Microsoft", "Technology", "Large"),
    "GOOGL": ("Alphabet", "Communication", "Large"), "META": ("Meta Platforms", "Communication", "Large"),
    "NVDA": ("NVIDIA", "Technology", "Large"), "AMD": ("AMD", "Technology", "Large"),
    "AVGO": ("Broadcom", "Technology", "Large"), "ORCL": ("Oracle", "Technology", "Large"),
    "CRM": ("Salesforce", "Technology", "Large"), "ADBE": ("Adobe", "Technology", "Large"),
    "INTC": ("Intel", "Technology", "Large"), "CSCO": ("Cisco", "Technology", "Large"),
    "NFLX": ("Netflix", "Communication", "Large"),
    # ---- Small & mid-size companies (so the size tilt has something to work with)
    "SFM": ("Sprouts Farmers Market", "Consumer Staples", "Small/Mid"),
    "ELF": ("e.l.f. Beauty", "Consumer Staples", "Small/Mid"),
    "CAKE": ("Cheesecake Factory", "Consumer Discretionary", "Small/Mid"),
    "SHAK": ("Shake Shack", "Consumer Discretionary", "Small/Mid"),
    "BOOT": ("Boot Barn", "Consumer Discretionary", "Small/Mid"),
    "AEO": ("American Eagle Outfitters", "Consumer Discretionary", "Small/Mid"),
    "M": ("Macy's", "Consumer Discretionary", "Small/Mid"),
    "KSS": ("Kohl's", "Consumer Discretionary", "Small/Mid"),
    "KBH": ("KB Home", "Consumer Discretionary", "Small/Mid"),
    "MTH": ("Meritage Homes", "Consumer Discretionary", "Small/Mid"),
    "WGO": ("Winnebago", "Consumer Discretionary", "Small/Mid"),
    "RMBS": ("Rambus", "Technology", "Small/Mid"),
    "AAON": ("AAON", "Industrials", "Small/Mid"),
    "ZION": ("Zions Bancorp", "Financials", "Small/Mid"),
    "FHN": ("First Horizon", "Financials", "Small/Mid"),
    "WAL": ("Western Alliance", "Financials", "Small/Mid"),
    "RUN": ("Sunrun", "Utilities", "Small/Mid"),
    "VSAT": ("Viasat", "Communication", "Small/Mid"),
}

# Example portfolio pre-filled in Tab 2 (tech-heavy, so the app has something to fix)
EXAMPLE_HOLDINGS = [("AAPL", 3000.0), ("MSFT", 2500.0), ("NVDA", 2000.0), ("TSLA", 1500.0), ("JPM", 1000.0)]


def company(t):
    if t == CASH:
        return CASH_NAME
    return UNIVERSE.get(t, (t, "Your stock", "—"))[0]


def sector(t):
    if t == CASH:
        return "Cash"
    return UNIVERSE.get(t, (t, "Your stock", "—"))[1]


# =============================================================================
# 3. DATA
# =============================================================================

@st.cache_data(ttl=6 * 3600, show_spinner="Downloading stock prices from Yahoo Finance...")
def download_prices(tickers):
    """Monthly adjusted closing prices (dividends and splits included), last 11 years.
    Index = calendar month, so it lines up with the Fama-French months."""
    try:
        raw = yf.download(list(tickers), period="11y", interval="1mo",
                          auto_adjust=True, progress=False)
        close = raw["Close"]
    except Exception:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.to_period("M")
    close = close[~close.index.duplicated(keep="last")]
    return close.dropna(axis=1, how="all")


@st.cache_data(ttl=6 * 3600)
def current_risk_free_rate():
    """Today's risk-free rate: 13-week T-bill yield (^IRX). Yahoo reports 4.2 for 4.2%."""
    try:
        irx = yf.download("^IRX", period="1mo", progress=False, auto_adjust=True)["Close"].dropna()
        return float(np.ravel(irx.values)[-1]) / 100
    except Exception:
        return None


def parse_ff_csv(text):
    """Ken French CSV: a few description lines, then monthly rows like
    '192607,    2.89,   -2.55,   -2.39,    0.22' (in percent), then an annual section."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts[:5])
        elif rows and "Annual" in line:
            break
    df = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    df.index = pd.PeriodIndex(pd.to_datetime(df.pop("date"), format="%Y%m"), freq="M")
    return df.astype(float) / 100


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading Fama-French factors...")
def load_fama_french():
    try:
        req = urllib.request.Request(FF_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        return parse_ff_csv(text), "Kenneth French Data Library (downloaded live)"
    except Exception:
        try:
            with open(FF_LOCAL_FILE, encoding="latin-1") as f:
                return parse_ff_csv(f.read()), f"Kenneth French Data Library (local file {FF_LOCAL_FILE})"
        except Exception:
            return None, None


# =============================================================================
# 4. MODELS: CAPM + FAMA-FRENCH REGRESSIONS, EXPECTED RETURNS, RISK
# =============================================================================

@st.cache_data(ttl=6 * 3600, show_spinner="Estimating CAPM and Fama-French models...")
def build_model(window_years, extra_tickers=()):
    """For every stock:
       CAPM regression:  R - RF = a + b*(Mkt-RF)                    -> alpha (a) and beta (b)
       FF3 regression:   R - RF = a + b*(Mkt-RF) + s*SMB + h*HML    -> market, size, value loadings
    Then expected returns, volatility, stock Sharpe and the covariance matrix."""
    ff, ff_source = load_fama_french()
    if ff is None:
        return {"error": "Couldn't download the Fama-French factors. Download "
                         "'F-F_Research_Data_Factors_CSV.zip' from the Ken French Data Library, "
                         f"unzip it, and put {FF_LOCAL_FILE} in the same folder as app.py."}

    tickers = tuple(sorted(set(UNIVERSE) | set(extra_tickers)))
    prices = download_prices(tickers)
    if prices.empty:
        return {"error": "Couldn't download prices from Yahoo Finance. Wait a minute and refresh."}
    returns = prices.pct_change(fill_method=None)

    # Use the months where we have BOTH stock returns and factor data
    # (the factor file is usually 1-2 months behind)
    common = returns.index.intersection(ff.index)
    months = common[-window_years * 12:]
    R = returns.loc[months]
    F = ff.loc[months]

    # keep only stocks with a full history in the window
    complete = [t for t in R.columns if R[t].notna().all()]
    missing = sorted(set(tickers) - set(complete))
    R = R[complete]
    excess = R.sub(F["RF"], axis=0)

    X_capm = F[["Mkt-RF"]].values
    X_ff = F[["Mkt-RF", "SMB", "HML"]].values
    rows = []
    for t in complete:
        y = excess[t].values
        capm = LinearRegression().fit(X_capm, y)
        ff3 = LinearRegression().fit(X_ff, y)
        raw_beta = capm.coef_[0]
        rows.append({
            "Ticker": t, "Company": company(t), "Sector": sector(t),
            "Beta (raw)": raw_beta,
            "Beta (adj)": BLUME_WEIGHT * raw_beta + (1 - BLUME_WEIGHT) * 1.0,
            # alpha = average monthly return beyond what CAPM predicts, x12 = per year.
            # SHOWN in the app, NOT used to pick stocks (past alpha rarely repeats, Lectures 17-18)
            "Alpha": capm.intercept_ * 12,
            "CAPM R2": capm.score(X_capm, y),
            "FF Mkt": ff3.coef_[0], "FF SMB": ff3.coef_[1], "FF HML": ff3.coef_[2],
            "FF R2": ff3.score(X_ff, y),
        })
    stats = pd.DataFrame(rows).set_index("Ticker")

    # Factor premiums: long-run annual averages since July 1963
    hist = ff.loc[ff.index >= pd.Period(PREMIUM_START, "M")]
    prem = {"MKT": hist["Mkt-RF"].mean() * 12, "SMB": hist["SMB"].mean() * 12,
            "HML": hist["HML"].mean() * 12}

    rf = current_risk_free_rate()
    rf_source = "13-week U.S. Treasury bill yield (^IRX, Yahoo Finance)"
    if rf is None:
        rf = float(ff["RF"].iloc[-1] * 12)
        rf_source = "latest 1-month T-bill rate in the Fama-French file, annualized"

    # Expected returns (we assume alpha = 0 going forward: past alpha rarely repeats)
    stats["CAPM E[R]"] = rf + stats["Beta (adj)"] * prem["MKT"]
    stats["FF E[R]"] = (rf + stats["FF Mkt"] * prem["MKT"] + stats["FF SMB"] * prem["SMB"]
                        + stats["FF HML"] * prem["HML"])
    stats["Volatility"] = R.std() * np.sqrt(12)
    stats["Sharpe"] = (stats["FF E[R]"] - rf) / stats["Volatility"]

    # Risk: covariance of monthly returns with Ledoit-Wolf shrinkage (less noisy than the
    # raw sample covariance when we have many stocks and few months), annualized x12
    cov = pd.DataFrame(LedoitWolf().fit(R.values).covariance_ * 12, index=complete, columns=complete)

    market_vol = F["Mkt-RF"].std() * np.sqrt(12)
    return {
        "stats": stats, "cov": cov, "rf": rf, "rf_source": rf_source, "prem": prem,
        "ff_source": ff_source, "start": str(months[0]), "end": str(months[-1]),
        "n_months": len(months), "missing": missing,
        "market_vol": market_vol, "market_sharpe": prem["MKT"] / market_vol,
    }


# =============================================================================
# 5. PORTFOLIO MATH
# =============================================================================

def _arrays(tickers, model):
    s = model["stats"].loc[list(tickers)]
    cov = model["cov"].loc[list(tickers), list(tickers)].values
    return (s["Beta (adj)"].values, s["FF SMB"].values, s["FF HML"].values,
            s["FF E[R]"].values, cov)


def port_metrics(w, tickers, model):
    """w = stock weights as a share of the WHOLE portfolio. Whatever is left over
    (1 - sum of w) is in T-bills: beta 0, factor loadings 0, return = risk-free rate,
    no volatility (Lecture 12: risk-free investments have beta = 0).
    Every portfolio number is a weighted average of the holdings' numbers,
    except volatility, which uses the covariance matrix: sqrt(w' Cov w)."""
    w = np.asarray(w, dtype=float)
    cash = max(0.0, 1.0 - float(w.sum()))
    s = model["stats"].loc[list(tickers)]
    cov = model["cov"].loc[list(tickers), list(tickers)].values
    rf = model["rf"]
    vol = float(np.sqrt(w @ cov @ w))
    ff_er = float(w @ s["FF E[R]"].values) + cash * rf
    return {
        "beta": float(w @ s["Beta (adj)"].values),
        "mkt": float(w @ s["FF Mkt"].values),
        "smb": float(w @ s["FF SMB"].values),
        "hml": float(w @ s["FF HML"].values),
        "alpha": float(w @ s["Alpha"].values),
        "capm_er": float(w @ s["CAPM E[R]"].values) + cash * rf,
        "ff_er": ff_er,
        "excess": ff_er - rf,
        "vol": vol,
        "sharpe": (ff_er - rf) / vol,
        "cash": cash,
    }


def beta_range(betas, w_min, w_max):
    """Lowest and highest portfolio beta these stocks can reach within the position limits."""
    n = len(betas)
    bounds = [(w_min, w_max)] * n
    lo = linprog(betas, A_eq=[np.ones(n)], b_eq=[1], bounds=bounds).fun
    hi = -linprog(-betas, A_eq=[np.ones(n)], b_eq=[1], bounds=bounds).fun
    return lo, hi


def recommended_limits(n):
    """No position smaller than half, or bigger than double, an equal share (1/n).
    Keeps the portfolio diversified while letting the optimizer tilt. Rounded to 0.5%."""
    w_min = min(0.5 / n, 0.20)
    w_max = min(1.0, max(2.0 / n, 0.05))
    return round(w_min * 200) / 200, round(w_max * 200) / 200


def limits_ok(n, w_min, w_max):
    return w_min <= w_max and n * w_min <= 1 + 1e-9 and n * w_max >= 1 - 1e-9


def to_dollars(weights, amount):
    """Weights -> dollars, with rounding fixed so the total is exact."""
    d = np.round(np.asarray(weights) * amount, 2)
    d[np.argmax(d)] += round(amount - d.sum(), 2)
    return d


def to_pct(weights):
    """Weights -> percentages with 1 decimal that add up to exactly 100.0."""
    p = np.round(np.asarray(weights, dtype=float) * 100, 1)
    if p.sum() > 0:
        p[np.argmax(p)] = round(p[np.argmax(p)] + 100 - p.sum(), 1)
    return p


# =============================================================================
# 6. "START FROM SCRATCH" LOGIC: AGE -> TARGET BETA -> STOCKS -> CAPITAL ALLOCATION LINE
# =============================================================================

def age_risk_band(age):
    """(beta floor, beta ceiling) by age. Our assumption, based on the idea behind the
    '110 - age = % in stocks' rule and target-date funds: younger investors have more
    years to recover from a market drop, so they can take more market risk."""
    if age < 30:
        return 0.90, 1.50
    elif age < 45:
        return 0.80, 1.30
    elif age < 60:
        return 0.60, 1.10
    else:
        return 0.40, 0.90


def target_beta_from_slider(age, risk_pct):
    """Risk slider 0-100 = position inside the age range (0 = floor, 100 = ceiling)."""
    floor, ceiling = age_risk_band(age)
    return floor + (risk_pct / 100) * (ceiling - floor)


def pick_beginner_stocks(model, target_beta, n, seed):
    """Step 1: candidate pool = the 2N stocks whose beta is closest to the target.
    Step 2: keep the N with the best stock Sharpe (FF expected excess return / volatility).
    Shuffle (seed > 0): pick N at random from the same pool instead."""
    s = model["stats"]
    pool_size = min(len(s), 2 * n)
    pool = (s["Beta (adj)"] - target_beta).abs().nsmallest(pool_size).index.tolist()
    if seed == 0:
        picks = s.loc[pool, "Sharpe"].nlargest(n).index.tolist()
    else:
        rng = np.random.default_rng(seed)
        picks = [str(t) for t in rng.choice(pool, size=n, replace=False)]
    return picks, pool


def max_sharpe(tickers, model, target_beta, w_min, w_max):
    """All-stock weights that MAXIMIZE the Sharpe ratio subject to:
       weights add to 100%, each weight within [w_min, w_max], portfolio beta = target."""
    b, _, _, mu, cov = _arrays(tickers, model)
    rf, n = model["rf"], len(tickers)
    bounds = [(w_min, w_max)] * n

    lo, hi = beta_range(b, w_min, w_max)
    tb = float(np.clip(target_beta, lo, hi))   # if the target can't be reached, use the closest

    # feasible starting point (hits the constraints exactly)
    start = linprog(np.zeros(n), A_eq=[np.ones(n), b], b_eq=[1, tb], bounds=bounds)
    w0 = start.x if start.success else np.ones(n) / n

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: w @ b - tb}]
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500})
    w = res.x if (abs(res.x.sum() - 1) < 1e-4 and abs(res.x @ b - tb) < 1e-3) else w0
    w = np.clip(w, w_min, w_max)
    return w / w.sum(), tb, (lo, hi)


def tangency(tickers, model, w_min, w_max):
    """The highest-Sharpe mix of these stocks, with NO beta target. On a risk/return chart,
    this is the point where the Capital Allocation Line (the line starting at T-bills)
    touches the best stock combinations."""
    _, _, _, mu, cov = _arrays(tickers, model)
    rf, n = model["rf"], len(tickers)

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=[(w_min, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500})
    w = res.x if abs(res.x.sum() - 1) < 1e-4 else w0
    w = np.clip(w, w_min, w_max)
    return w / w.sum()


def best_portfolio(tickers, model, target_beta, w_min, w_max, allow_cash):
    """CAPITAL ALLOCATION LINE (CAL) - how T-bills enter the portfolio:
      1. Find the highest-Sharpe stock mix (tangency).
      2. If your target beta is BELOW that mix's beta (and T-bills are allowed), keep the
         mix and put the rest in T-bills. T-bills have beta 0, so
             portfolio beta = stock share x mix beta  ->  stock share = target / mix beta.
         Risk goes down but the Sharpe ratio stays at its highest.
      3. If your target is ABOVE the mix's beta (or T-bills are off), stay 100% in stocks
         and choose the weights with the highest Sharpe at exactly your target beta.
         (We don't borrow money to go above the mix.)
    Position limits apply to the stock part. Returned weights are shares of the TOTAL
    portfolio (they add up to the stock share; the rest is T-bills)."""
    b = model["stats"].loc[list(tickers), "Beta (adj)"].values
    if allow_cash:
        w_t = tangency(tickers, model, w_min, w_max)
        mix_beta = float(w_t @ b)
        if target_beta < mix_beta * (1 - MIN_CASH):     # only worth it if T-bills would be >= 2%
            y = max(target_beta / mix_beta, 1 - MAX_CASH)
            return {"w": y * w_t, "cash": 1 - y, "mode": "cal",
                    "mix_beta": mix_beta, "beta_used": y * mix_beta}
    w, tb, _ = max_sharpe(tickers, model, target_beta, w_min, w_max)
    return {"w": w, "cash": 0.0, "mode": "stocks", "mix_beta": float(w @ b), "beta_used": tb}


# =============================================================================
# 7. "IMPROVE MY PORTFOLIO" LOGIC: ANALYZE -> RECOMMENDED GOALS -> REWEIGHT + SUBSTITUTES
# =============================================================================

def holdings_tickers(holdings):
    return tuple(sorted(t for t, _ in holdings if t != CASH))


def split_holdings(holdings, model):
    """-> (stocks we have data for, tickers we couldn't use, dollars in T-bills/cash)"""
    stats = model["stats"]
    good, bad, cash = [], [], 0.0
    for t, d in holdings:
        if t == CASH:
            cash += d
        elif t in stats.index:
            good.append((t, d))
        else:
            bad.append(t)
    return good, bad, cash


def current_weights(good, cash):
    dollars = np.array([d for _, d in good])
    total = dollars.sum() + cash
    return [t for t, _ in good], dollars, dollars / total, cash / total, total


def soft_optimize(tickers, base_w, base_c, targets, model, w_min, w_max, allow_cash):
    """Find weights that get as close as possible to ALL goals at once, with the
    smallest change from what you own now (base_w = stocks, base_c = T-bills). Minimizes:
        50*(beta - target)^2 + 20*(size - target)^2 + 20*(value - target)^2
      + 50*(Sharpe shortfall below target)^2 + 1*(sum of squared weight changes)
    Goals are 'soft', so the optimizer always returns an answer even when a goal
    can't be fully reached; the app then reports how close it got.
    With T-bills allowed, the T-bill share c is one more decision (0 to 75%), and each
    stock must stay between w_min and w_max OF THE STOCK PART."""
    b, smb, hml, mu, cov = _arrays(tickers, model)
    rf, n = model["rf"], len(tickers)

    def score(w, c):
        sr = (w @ mu + c * rf - rf) / np.sqrt(w @ cov @ w)
        return (W_BETA * (w @ b - targets["beta"]) ** 2
                + W_TILT * (w @ smb - targets["smb"]) ** 2
                + W_TILT * (w @ hml - targets["hml"]) ** 2
                + W_SHARPE * max(0.0, targets["sharpe"] - sr) ** 2
                + W_TURN * (np.sum((w - base_w) ** 2) + (c - base_c) ** 2))

    if not allow_cash:
        start = np.clip(base_w / max(base_w.sum(), 1e-9), w_min, w_max)
        start = start / start.sum()
        res = minimize(lambda w: score(w, 0.0), start, method="SLSQP",
                       bounds=[(w_min, w_max)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 300})
        w = res.x if abs(res.x.sum() - 1) < 1e-4 else start
        w = np.clip(w, w_min, w_max)
        w = w / w.sum()
        return w, 0.0, float(score(w, 0.0))

    # decision vector x = [stock weights..., T-bill share]
    c0 = min(base_c, MAX_CASH)
    rel = np.clip(base_w / max(base_w.sum(), 1e-9), w_min, w_max)
    start = np.append(rel / rel.sum() * (1 - c0), c0)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
            {"type": "ineq", "fun": lambda x: x[:n] - w_min * (1 - x[n])},
            {"type": "ineq", "fun": lambda x: w_max * (1 - x[n]) - x[:n]}]
    res = minimize(lambda x: score(x[:n], x[n]), start, method="SLSQP",
                   bounds=[(0, w_max)] * n + [(0, MAX_CASH)], constraints=cons,
                   options={"maxiter": 300})
    x = res.x if abs(res.x.sum() - 1) < 1e-4 else start
    c = float(np.clip(x[n], 0, MAX_CASH))
    w = np.clip(x[:n], 0, None)
    w = w / w.sum() * (1 - c)
    return w, c, float(score(w, c))


@st.cache_data(show_spinner=False)
def recommended_settings(holdings, window, extra, allow_cash):
    """Default slider values for the Improve tab, each with a reason shown in the app.
    They come from the 'best version' of your portfolio: same stocks, best weights
    (plus T-bills if allowed) at your current beta."""
    model = build_model(window, extra)
    good, _, cash = split_holdings(holdings, model)
    tickers, _, w_cur, c_cur, _ = current_weights(good, cash)
    now = port_metrics(w_cur, tickers, model)

    w_min, w_max = recommended_limits(len(tickers))
    beta = round(float(np.clip(now["beta"], 0.3, 2.0)), 2)
    best = best_portfolio(tickers, model, beta, w_min, w_max, allow_cash)
    bm = port_metrics(best["w"], tickers, model)
    smb = round(float(np.clip(bm["smb"], -0.6, 1.0)), 2)    # kept inside the slider ranges
    hml = round(float(np.clip(bm["hml"], -0.8, 0.8)), 2)
    # check what the optimizer actually reaches with these goals, and recommend
    # that Sharpe (rounded down) so the default goals are always reachable
    goals = {"beta": beta, "smb": smb, "hml": hml, "sharpe": bm["sharpe"]}
    w_s, _, _ = soft_optimize(tickers, w_cur, c_cur, goals, model, w_min, w_max, allow_cash)
    reachable = port_metrics(w_s, tickers, model)["sharpe"]
    return {
        "beta": beta, "smb": smb, "hml": hml,
        "sharpe": float(np.clip(np.floor(reachable * 100) / 100, 0.0, 1.5)),
        "w_min": w_min, "w_max": w_max,
    }


def targets_met(m, targets):
    return {
        "beta": abs(m["beta"] - targets["beta"]) <= TOLERANCE["beta"],
        "smb": abs(m["smb"] - targets["smb"]) <= TOLERANCE["smb"],
        "hml": abs(m["hml"] - targets["hml"]) <= TOLERANCE["hml"],
        "sharpe": m["sharpe"] >= targets["sharpe"] - TOLERANCE["sharpe"],
    }


@st.cache_data(show_spinner="Finding the best adjustments to your portfolio...")
def recommend(holdings, window, extra, targets_tuple, w_min, w_max, max_swaps, seed, allow_cash):
    """Step 1: reweight the stocks you already own (and T-bills, if allowed).
    Step 2: if goals still aren't met, swap out your holding with the lowest stock Sharpe.
            Every stock in the universe is tried as its replacement (weights re-optimized
            each time); the one that gets closest to your goals wins. If no replacement
            helps, the next-weakest holding is tried (up to 3).
            Shuffle (seed > 0): choose at random among the 5 best replacements instead.
    Repeats step 2 up to max_swaps times, stopping early once goals are met."""
    model = build_model(window, extra)
    stats = model["stats"]
    good, _, cash = split_holdings(holdings, model)
    original, dollars, w_cur, c_cur, total = current_weights(good, cash)
    targets = dict(zip(["beta", "sharpe", "smb", "hml"], targets_tuple))

    current, base = list(original), w_cur.copy()
    w, c, score = soft_optimize(current, base, c_cur, targets, model, w_min, w_max, allow_cash)
    reweight_metrics = port_metrics(w, current, model)
    rng = np.random.default_rng(seed)
    swaps, removed = [], set()

    for _ in range(max_swaps):
        if all(targets_met(port_metrics(w, current, model), targets).values()):
            break
        # weakest first; if replacing it doesn't help, try the next weakest (up to 3)
        candidates_out = sorted([t for t in current if t in original], key=lambda t: stats.loc[t, "Sharpe"])
        improving = []
        for rank, weakest in enumerate(candidates_out[:3], start=1):
            i = current.index(weakest)
            sell_penalty = W_TURN * base[i] ** 2      # selling it all counts as turnover
            new_base = base.copy()
            new_base[i] = 0.0                          # new stock starts from 0%
            trials = []
            for cand_t in stats.index:
                if cand_t in current or cand_t in removed:
                    continue
                cand = list(current)
                cand[i] = cand_t
                wn, cn, sn = soft_optimize(cand, new_base, c_cur, targets, model, w_min, w_max, allow_cash)
                trials.append((sn + sell_penalty, cand_t, wn, cn))
            improving = sorted([t for t in trials if t[0] < score - 1e-4], key=lambda x: x[0])
            if improving:
                break
        if not improving:
            break

        k = 0 if seed == 0 else int(rng.integers(min(5, len(improving))))
        new_score, new_t, wn, cn = improving[k]
        swaps.append({"out": weakest, "in": new_t, "rank": rank,
                      "alternatives": [t[1] for t in improving[:6] if t[1] != new_t][:3]})
        removed.add(weakest)
        current[i] = new_t
        base = new_base
        w, c, score = wn, cn, new_score

    return {
        "original": original, "w_cur": w_cur, "c_cur": c_cur, "total": float(total),
        "dollars_cur": dollars, "cash_cur": float(cash),
        "final": current, "w_final": w, "c_final": c, "swaps": swaps,
        "before": port_metrics(w_cur, original, model),
        "reweight": reweight_metrics,
        "after": port_metrics(w, current, model),
    }


# =============================================================================
# 8. DISPLAY HELPERS
# =============================================================================

def explain(text):
    st.caption(f"💡 {text}")


def metric_grid(items, per_row=2):
    """Metrics in rows of 2: readable on a laptop, and each row stacks on a phone.
    items = [(label, value, help text), ...]"""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, item in zip(cols, items[i:i + per_row]):
            col.metric(item[0], item[1], help=item[2] if len(item) > 2 else None)


def size_word(smb):
    return "small-company" if smb > 0.25 else ("large-company" if smb < 0 else "mid-size")


def value_word(hml):
    return "value" if hml > 0.2 else ("growth" if hml < -0.2 else "blend")


def sharpe_band(x):
    for lo, hi, label in SHARPE_BANDS:
        if lo <= x < hi:
            return label
    return "Low"


def color_map(tickers):
    """Same stock = same color in every chart (alphabetical), T-bills always gray."""
    stocks = sorted(t for t in set(tickers) if t != CASH)
    cmap = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(stocks)}
    cmap[CASH] = CASH_COLOR
    return cmap


def sharpe_gauge(value, market_sharpe, title="Sharpe ratio", reference=None):
    """Gauge with rule-of-thumb color bands, an orange marker for the market's Sharpe,
    and (optionally) the change from a reference value."""
    top = max(1.2, value * 1.15, (reference or 0) * 1.15)
    colors = ["#f4cccc", "#fff2cc", "#d9ead3", "#b6d7a8"]
    steps = [{"range": [lo, min(hi, top)], "color": c} for (lo, hi, _), c in zip(SHARPE_BANDS, colors)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if reference is not None else ""),
        value=value,
        number={"valueformat": ".2f", "font": {"size": 40}},
        delta={"reference": reference, "valueformat": ".2f"} if reference is not None else None,
        title={"text": f"{title}: {sharpe_band(value)}", "font": {"size": 17}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, top]}, "bar": {"color": "#1f4e79", "thickness": 0.35},
               "steps": steps,
               "threshold": {"line": {"color": MARKET_COLOR, "width": 5}, "thickness": 0.9,
                             "value": market_sharpe}}))
    fig.add_annotation(text=f"<b>Orange line</b> = stock market's Sharpe ({market_sharpe:.2f})",
                       x=0.5, y=-0.12, xref="paper", yref="paper", showarrow=False,
                       font={"size": 12, "color": MARKET_COLOR})
    fig.update_layout(height=300, margin=dict(t=70, b=45, l=35, r=35))
    return fig


def sharpe_explanation(m, model):
    st.markdown(
        f"**Sharpe ratio = (expected return − risk-free rate) ÷ volatility**  \n"
        f"= ({m['ff_er']:.2%} − {model['rf']:.2%}) ÷ {m['vol']:.2%} = **{m['sharpe']:.2f}**")
    explain("It measures how much extra return you expect for each unit of risk. Higher is better. "
            f"The orange line is the overall stock market ({model['market_sharpe']:.2f}) "
            "using the same assumptions. The color bands are our rule-of-thumb labels.")


def return_waterfall(m, model):
    """Fama-French expected return, split into where it comes from."""
    p, rf = model["prem"], model["rf"]
    parts = [("Risk-free rate", rf), ("Market", m["mkt"] * p["MKT"]),
             ("Size (SMB)", m["smb"] * p["SMB"]), ("Value (HML)", m["hml"] * p["HML"])]
    fig = go.Figure(go.Waterfall(
        x=[x for x, _ in parts] + ["Expected return"],
        measure=["relative"] * len(parts) + ["total"],
        y=[v * 100 for _, v in parts] + [0],
        text=[f"{v:+.1%}" for _, v in parts] + [f"{m['ff_er']:.1%}"],
        textposition="outside", cliponaxis=False))
    fig.update_layout(height=320, margin=dict(t=30, b=10, l=10, r=10),
                      yaxis_title="% per year", showlegend=False)
    return fig


def capm_box(m, model):
    with st.container(border=True):
        st.markdown("#### CAPM (one factor: the market)")
        st.metric("CAPM expected return", f"{m['capm_er']:.2%}")
        st.markdown(f"E[R] = risk-free + beta × market premium  \n"
                    f"= {model['rf']:.2%} + {m['beta']:.2f} × {model['prem']['MKT']:.2%} "
                    f"= **{m['capm_er']:.2%}**")
        explain(f"**Beta {m['beta']:.2f}:** if the market moves 1%, this portfolio tends to move about "
                f"{m['beta']:.2f}%. Beta 1 = moves with the market; below 1 = calmer; above 1 = bigger swings. "
                "CAPM says the only risk you get paid for is market (systematic) risk, so return rises in a "
                "straight line with beta.")


def ff_box(m, model):
    p = model["prem"]
    with st.container(border=True):
        st.markdown("#### Fama-French 3-factor model")
        c1, c2, c3 = st.columns(3)
        c1.metric("Market", f"{m['mkt']:.2f}")
        c2.metric("Size (SMB)", f"{m['smb']:+.2f}")
        c3.metric("Value (HML)", f"{m['hml']:+.2f}")
        st.plotly_chart(return_waterfall(m, model))
        explain(f"**Market ({m['mkt']:.2f}):** sensitivity to the stock market, like beta. It can differ from the "
                "CAPM beta because it's estimated together with size and value, and it isn't Blume-adjusted.")
        explain(f"**Size / SMB ({m['smb']:+.2f}) → behaves like a {size_word(m['smb'])} portfolio.** "
                "SMB = Small Minus Big: positive means it moves like small companies, negative like large ones. "
                f"Small companies have historically earned {p['SMB']:.2%} a year more than large ones.")
        explain(f"**Value / HML ({m['hml']:+.2f}) → {value_word(m['hml'])} style.** "
                "HML = High Minus Low book-to-market: positive leans toward 'value' stocks (cheap relative to "
                "their book value), negative toward 'growth' stocks. "
                f"Value has historically earned {p['HML']:.2%} a year more than growth.")
        if m["smb"] * p["SMB"] + m["hml"] * p["HML"] > 0.02:
            explain(f"Size and value add **{m['smb'] * p['SMB'] + m['hml'] * p['HML']:.1%}** a year on top of "
                    "CAPM. That's why the Fama-French expected return is higher than the CAPM one.")


def alpha_explanation(model, window_years):
    explain(f"**Alpha** = how much more (or less) a stock returned per year than CAPM predicted, over the "
            f"last {window_years} years (Lecture 12). We **show** it but **don't use it to pick stocks**: "
            "past alpha rarely repeats and may just be luck (Lectures 17–18), so our forecasts assume alpha = 0.")


def weights_pie(labels, weights, cmap):
    df = pd.DataFrame({"Holding": labels, "Company": [company(t) for t in labels],
                       "Weight %": to_pct(weights)})
    df = df[df["Weight %"] > 0]
    fig = px.pie(df, names="Holding", values="Weight %", color="Holding",
                 color_discrete_map=cmap, hover_data=["Company"],
                 category_orders={"Holding": sorted(df["Holding"], key=lambda t: (t == CASH, t))})
    fig.update_traces(textposition="inside", textinfo="label+percent", sort=False)
    fig.update_layout(height=400, margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=-0.05))
    return fig


def compare_bars(labels, now_pct, rec_pct):
    """Now vs. recommended weight for every holding (easier to read than two pies, also on a phone)."""
    rows = []
    for t in labels:
        rows.append({"Holding": t, "Version": "Now", "Weight %": now_pct.get(t, 0.0)})
        rows.append({"Holding": t, "Version": "Recommended", "Weight %": rec_pct.get(t, 0.0)})
    df = pd.DataFrame(rows)
    fig = px.bar(df, y="Holding", x="Weight %", color="Version", barmode="group", orientation="h",
                 color_discrete_map={"Now": "#9AA5B1", "Recommended": "#1F77B4"},
                 category_orders={"Holding": labels, "Version": ["Now", "Recommended"]},
                 text="Weight %")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(height=120 + 55 * len(labels), margin=dict(t=10, b=10, l=10, r=40),
                      legend=dict(orientation="h", y=1.08, title=None), xaxis_title="% of portfolio")
    return fig


# =============================================================================
# 9. HOME PAGE
# =============================================================================

def home_page():
    st.subheader("Welcome! Your automated financial advisor")
    st.write("Not everyone has access to a financial advisor. This app does part of an advisor's job: "
             "it suggests **which stocks to hold, how much of each, and how much risk to take**, based on "
             "your age and how comfortable you are with risk. Every number comes with a plain-English explanation.")

    st.markdown("#### Which tab should I use?")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**🌱 Start from scratch**")
            st.write("New to investing? Enter your age, risk tolerance, amount and number of stocks. "
                     "We build a diversified portfolio and explain why we chose each stock.")
    with c2:
        with st.container(border=True):
            st.markdown("**📈 Improve my portfolio**")
            st.write("Already invest? Enter your stocks and dollar amounts. We score your portfolio and "
                     "recommend weight changes and substitute stocks to reach your goals.")
    st.caption("👆 Click a tab at the top of the page to get started.")

    st.markdown("#### How it works")
    st.markdown("""
1. **Measure risk.** For each stock we run two regressions on monthly returns: **CAPM** (market risk, *beta*) and the **Fama-French 3-factor model** (market, size and value).
2. **Estimate returns.** Expected return = risk-free rate + each risk exposure × that factor's long-run reward.
3. **Match your risk.** Your age and risk slider set a *target beta*. If you want less risk than the best stock mix, we add **T-bills** (beta 0).
4. **Optimize.** We choose weights with the highest **Sharpe ratio** (return per unit of risk) at your risk level.
""")

    st.markdown("#### Key terms")
    with st.expander("Beta, Sharpe ratio, Fama-French, alpha, T-bills"):
        st.markdown("""
- **Beta (β):** how much a stock or portfolio moves with the stock market. 1 = same as the market, below 1 = calmer, above 1 = bigger swings. It measures *systematic* risk, which diversification can't remove.
- **Volatility:** how much returns swing in a typical year (standard deviation). Includes *idiosyncratic* (company-specific) risk, which diversification reduces.
- **Sharpe ratio:** (expected return − risk-free rate) ÷ volatility. Extra return per unit of risk; portfolio managers try to maximize it.
- **Fama-French factors:** **SMB** (Small Minus Big): small companies vs. large; **HML** (High Minus Low book-to-market): value stocks vs. growth stocks.
- **Alpha (α):** return beyond what the model predicts. We show past alpha but don't use it to forecast.
- **T-bills:** short-term U.S. government debt. Beta 0, earns the risk-free rate. You can buy them through a T-bill ETF such as SGOV or BIL.
""")
    methods_section()
    st.caption("AC420 class project, not investment advice.")


def methods_section():
    model = build_model(DEFAULT_WINDOW_YEARS)
    if "error" in model:
        st.error(model["error"])
        return
    p = model["prem"]
    with st.expander("📘 Methods, data and assumptions"):
        st.markdown(f"""
**Data sources**
| What | Source |
|---|---|
| Stock prices | Yahoo Finance via `yfinance`: monthly adjusted closes (include dividends and splits) |
| Fama-French factors (Mkt-RF, SMB, HML, RF) | {model['ff_source']}, the same data WRDS distributes |
| Current risk-free rate | {model['rf_source']}: **{model['rf']:.2%}** |
| Estimation window | {model['start']} to {model['end']} ({model['n_months']} months) in "Start from scratch"; 3, 5 or 10 years in "Improve my portfolio" |

**Models (linear regressions, one per stock, on monthly excess returns)**
- **CAPM:** R − RF = α + β(Mkt−RF). β is then **Blume-adjusted**: 0.67 × β + 0.33, because betas drift toward 1 over time (Blume, 1971).
- **Fama-French 3-factor:** R − RF = α + b(Mkt−RF) + s·SMB + h·HML.
- **Expected returns:** CAPM: RF + β × market premium. Fama-French: RF + b × market premium + s × SMB premium + h × HML premium. We assume α = 0 going forward, because past α rarely repeats (Lectures 17–18). Past α is shown, not used.
- **Factor premiums** = average annual factor return since July 1963 (the start of Fama & French's 1993 sample): market **{p['MKT']:.2%}**, SMB **{p['SMB']:.2%}**, HML **{p['HML']:.2%}**.
- **Risk:** covariance of monthly returns with **Ledoit-Wolf shrinkage** (reduces noise when there are many stocks and few months), × 12 to annualize. Volatility = √(w′Σw).
- **Sharpe ratio** = (Fama-French expected return − RF) ÷ volatility. Market Sharpe for comparison: **{model['market_sharpe']:.2f}**.
- **We pick stocks by *expected* Sharpe (from the risk model), not *past* Sharpe**, because a high past Sharpe ratio is typically unrelated to the future one (Lecture 18).

**T-bills and the Capital Allocation Line**
T-bills have beta 0 and earn the risk-free rate. Mixing the best (highest-Sharpe) stock portfolio with T-bills lowers risk in a straight line: portfolio beta = stock share × stock beta, and the Sharpe ratio stays the same. So when your target beta is below the best stock mix's beta, we hold that mix plus T-bills (at most {MAX_CASH:.0%} in T-bills). When your target is above it, we stay 100% in stocks at your target beta (no borrowing).

**"Start from scratch" rules (our assumptions)**
| Age | Beta range |
|---|---|
| under 30 | 0.90 – 1.50 |
| 30–44 | 0.80 – 1.30 |
| 45–59 | 0.60 – 1.10 |
| 60+ | 0.40 – 0.90 |

Based on the idea behind the "110 − age = % in stocks" rule and target-date funds: more years until you need the money = more risk you can recover from. The risk slider picks a point in the range. Stocks: the 2N closest in beta, then the N best expected stock Sharpes. Each stock is between ½ and 2× an equal share of the stock part.

**"Improve my portfolio" rules**
Recommended goals come from the **highest-Sharpe version of your current portfolio** (same stocks, plus T-bills if allowed, at today's beta). Optimizer: minimizes 50·(beta gap)² + 20·(size gap)² + 20·(value gap)² + 50·(Sharpe shortfall)² + 1·(weight changes)². Substitutes replace the holding with the lowest expected stock Sharpe.

**Limitations:** estimates come from past data and may not hold in the future. The universe is {len(model['stats'])} stocks that exist today (survivorship bias). Bonds aren't included, since CAPM and Fama-French are stock models. Taxes and trading costs are ignored. The Fama-French file is published with a 1–2 month delay.
""")
        if model["missing"]:
            st.caption("Stocks in our list skipped for missing data: " + ", ".join(model["missing"]))


# =============================================================================
# 10. "START FROM SCRATCH" PAGE
# =============================================================================

def beginner_tab():
    st.subheader("Build your first portfolio")
    st.write("Answer a few questions and we'll build a diversified portfolio for you.")

    model = build_model(DEFAULT_WINDOW_YEARS)
    if "error" in model:
        st.error(model["error"])
        return

    # ---------- inputs
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Your age", min_value=18, max_value=100, value=22, step=1, key="beg_age")
            floor, ceiling = age_risk_band(age)
            explain(f"Your age sets the range of risk we allow: **beta {floor:.2f} to {ceiling:.2f}**. "
                    "Younger investors have more years to recover from a market drop.")
            risk = st.slider("Risk tolerance (within your age range)", 0, 100, 50, key="beg_risk")
            target = target_beta_from_slider(age, risk)
            explain(f"0 = safest end of your range, 100 = riskiest. Your **target beta is {target:.2f}** "
                    "(1.0 = moves like the stock market).")
        with c2:
            amount = st.number_input("Amount to invest ($)", min_value=100, value=10_000, step=500, key="beg_amount")
            explain("We split this amount across your holdings.")
            n = st.slider("Number of stocks", 10, 30, 15, key="beg_n")
            explain("More stocks = more diversification, so one company having a bad year hurts you less.")
        allow_cash = st.checkbox("Use T-bills (cash) to lower risk when needed", value=True, key="beg_cash")
        explain("T-bills have beta 0. If your target risk is below the best stock mix, we add T-bills instead of "
                "settling for weaker stocks. Turn this off to stay 100% in stocks.")

    # ---------- shuffle controls
    st.session_state.setdefault("beg_seed", 0)
    b1, b2 = st.columns(2)
    if b1.button("🔀 Shuffle stocks", key="beg_shuffle"):
        st.session_state["beg_seed"] += 1
    if b2.button("↩️ Back to best picks", key="beg_reset"):
        st.session_state["beg_seed"] = 0
    seed = st.session_state["beg_seed"]

    # ---------- model
    picks, pool = pick_beginner_stocks(model, target, n, seed)
    w_min, w_max = recommended_limits(n)
    best = best_portfolio(picks, model, target, w_min, w_max, allow_cash)
    w, cash = best["w"], best["cash"]
    m = port_metrics(w, picks, model)

    if seed > 0:
        st.info(f"Shuffle #{seed}: these {n} stocks were drawn at random from the {len(pool)} stocks closest "
                "to your target beta. Shuffled portfolios usually score a little lower on the Sharpe gauge "
                "than the best picks. Compare them!")
    if best["mode"] == "cal":
        msg = (f"**Why T-bills?** Your target beta ({target:.2f}) is below the beta of the best stock mix "
               f"({best['mix_beta']:.2f}). So we put **{1 - cash:.0%} in that mix and {cash:.0%} in T-bills**. "
               f"T-bills have beta 0, so portfolio beta = {1 - cash:.2f} × {best['mix_beta']:.2f} = **{m['beta']:.2f}**. "
               "This is the Capital Allocation Line: less risk, same (highest) Sharpe ratio.")
        if abs(m["beta"] - target) > 0.005:
            msg += f" (We cap T-bills at {MAX_CASH:.0%}, so beta stays at {m['beta']:.2f}.)"
        st.info(msg)
    elif abs(best["beta_used"] - target) > 0.005:
        hint = "" if allow_cash else " Turn on T-bills to reach a lower risk level."
        st.info(f"A beta of {target:.2f} isn't reachable with these stocks, so we used the closest possible "
                f"({best['beta_used']:.2f}).{hint}")

    # ---------- headline numbers
    st.markdown("### Your portfolio")
    metric_grid([
        ("Risk (beta)", f"{m['beta']:.2f}", "How much the portfolio moves with the market. 1.0 = same as the market."),
        ("Expected return", f"{m['ff_er']:.1%}", "Fama-French estimate of the average yearly return."),
        ("Volatility", f"{m['vol']:.1%}", "How much returns typically swing in a year (standard deviation)."),
        ("Sharpe ratio", f"{m['sharpe']:.2f}", "Return per unit of risk. Higher is better."),
    ])

    st.plotly_chart(sharpe_gauge(m["sharpe"], model["market_sharpe"]))
    sharpe_explanation(m, model)

    labels = picks + ([CASH] if cash > 1e-6 else [])
    weights = np.append(w, cash) if cash > 1e-6 else w
    cmap = color_map(labels)
    st.markdown("### Where your money goes")
    st.plotly_chart(weights_pie(labels, weights, cmap))

    # ---------- holdings table with reasons
    st.markdown("### Your holdings and why we chose them")
    s = model["stats"]
    pool_rank = s.loc[pool, "Sharpe"].rank(ascending=False).astype(int)
    dollars = to_dollars(weights, amount)
    pcts = to_pct(weights)
    rows = []
    for t, pct, d in zip(labels, pcts, dollars):
        if t == CASH:
            rows.append({"Ticker": CASH, "Company": CASH_NAME, "Weight %": pct, "$ Amount": d,
                         "Beta": 0.0, "Alpha": None, "Exp. return": round(model["rf"] * 100, 1),
                         "Why chosen": "Beta 0, earns the risk-free rate. Lowers risk to your target without "
                                       "lowering the Sharpe ratio (Capital Allocation Line)."})
            continue
        r = s.loc[t]
        if seed == 0:
            why = (f"Beta {r['Beta (adj)']:.2f} is close to your target; ranked #{pool_rank[t]} of "
                   f"{len(pool)} candidates on expected return per unit of risk (stock Sharpe {r['Sharpe']:.2f}).")
        else:
            why = (f"Beta {r['Beta (adj)']:.2f} is close to your target; drawn at random from the "
                   f"{len(pool)} closest-beta stocks.")
        rows.append({"Ticker": t, "Company": r["Company"], "Weight %": pct, "$ Amount": d,
                     "Beta": round(r["Beta (adj)"], 2), "Alpha": round(r["Alpha"] * 100, 1),
                     "Exp. return": round(r["FF E[R]"] * 100, 1),
                     "Why chosen": f"{why} {r['Sector']}; {size_word(r['FF SMB'])}, {value_word(r['FF HML'])}."})
    table = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
    st.dataframe(table, hide_index=True, column_config={
        "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
        "$ Amount": st.column_config.NumberColumn(format="$%.2f"),
        "Alpha": st.column_config.NumberColumn("Past alpha", format="%.1f%%"),
        "Exp. return": st.column_config.NumberColumn(format="%.1f%%")})
    st.write(f"**Total:** {pcts.sum():.1f}% · ${dollars.sum():,.2f}")
    alpha_explanation(model, DEFAULT_WINDOW_YEARS)

    # ---------- CAPM and Fama-French
    st.markdown("### What the models say")
    capm_box(m, model)
    ff_box(m, model)
    explain("We show both models. Fama-French is the one we use to pick stocks and set weights, "
            "because it adds size and value to the market factor and explains returns better than CAPM alone (Lecture 13).")

    # ---------- how the stocks were picked
    with st.expander("🔍 How did the model pick these stocks?"):
        cal_step = (f"6. **T-bills:** the best stock mix has beta {best['mix_beta']:.2f}, above your target "
                    f"{target:.2f}, so we hold {1 - cash:.0%} in it and {cash:.0%} in T-bills (Capital Allocation Line)."
                    if best["mode"] == "cal" else
                    "6. **T-bills:** not needed. Your target beta is at or above the best stock mix's beta, so "
                    "you're 100% in stocks with the highest Sharpe at your target beta.")
        st.markdown(f"""
1. **Universe:** {len(s)} U.S. stocks across every sector, large and small/mid-size (full list below).
2. **Your target beta:** your age range ({floor:.2f}–{ceiling:.2f}) and risk slider ({risk}) give **{target:.2f}**.
3. **Candidate pool:** the **{len(pool)} stocks** (2 × {n}) whose beta is closest to {target:.2f}.
4. **Pick {n}:** the {n} with the highest *expected* stock Sharpe ratio (Fama-French expected return above the risk-free rate ÷ the stock's volatility). **Shuffle** draws {n} at random from the pool instead.
5. **Weights:** an optimizer finds the stock weights with the **highest Sharpe ratio**, each between **{w_min:.1%} and {w_max:.1%}** of the stock part (half to double an equal share).
{cal_step}
""")
        pool_df = s.loc[pool, ["Company", "Sector", "Beta (adj)", "Alpha", "FF E[R]", "Volatility", "Sharpe"]].copy()
        pool_df.insert(0, "Selected", ["✅" if t in picks else "" for t in pool_df.index])
        st.dataframe(pool_df.sort_values("Sharpe", ascending=False).round(3))


# =============================================================================
# 11. "IMPROVE MY PORTFOLIO" PAGE
# =============================================================================

def clean_holdings(df):
    d = df.copy()
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    d.loc[d["Ticker"].isin(CASH_ALIASES), "Ticker"] = CASH
    d["$ Amount"] = pd.to_numeric(d["$ Amount"], errors="coerce")
    d = d[~d["Ticker"].isin(["", "NONE", "NAN"]) & (d["$ Amount"] > 0)]
    d = d.groupby("Ticker", sort=False)["$ Amount"].sum()
    return tuple((t, float(v)) for t, v in d.items())


def action_label(cur, new, total):
    if cur == 0:
        return "🟢 New buy"
    if new == 0:
        return "🔴 Sell all"
    diff = new - cur
    if diff > 0.01 * total:
        return "⬆️ Buy more"
    if diff < -0.01 * total:
        return "⬇️ Trim"
    return "Hold"


def existing_tab():
    st.subheader("Improve a portfolio you already own")
    st.write("Enter your holdings, see how your portfolio scores on CAPM, Fama-French and the Sharpe ratio, "
             "then set your goals. We recommend weight changes and substitute stocks to reach them.")

    st.session_state.setdefault("exp_holdings", tuple(EXAMPLE_HOLDINGS))
    st.session_state.setdefault("exp_window", DEFAULT_WINDOW_YEARS)
    st.session_state.setdefault("exp_seed", 0)

    # ---------- 1. holdings input
    st.markdown("#### 1. Your current portfolio")
    with st.form("holdings_form"):
        edited = st.data_editor(
            pd.DataFrame(EXAMPLE_HOLDINGS, columns=["Ticker", "$ Amount"]),
            num_rows="dynamic", key="exp_editor",
            column_config={"Ticker": st.column_config.TextColumn(help="Stock symbol, e.g. AAPL. Type CASH for T-bills or cash."),
                           "$ Amount": st.column_config.NumberColumn(min_value=0, format="$%.2f")})
        explain("Type each stock's ticker and how many dollars you have in it; add rows with the + at the bottom. "
                "Any U.S. stock on Yahoo Finance works. Type **CASH** for money in T-bills or a money market fund.")
        window = st.select_slider("History window", options=WINDOW_OPTIONS, value=DEFAULT_WINDOW_YEARS,
                                  format_func=lambda y: f"{y} years")
        explain("How many years of monthly returns we use for the regressions. **We recommend 5 years** "
                "(60 months), the most common choice for beta. 3 years reacts faster but is noisier; "
                "10 years is more stable but includes older data.")
        if st.form_submit_button("Analyze my portfolio", type="primary"):
            st.session_state["exp_holdings"] = clean_holdings(edited)
            st.session_state["exp_window"] = window
            st.session_state["exp_seed"] = 0

    holdings = st.session_state["exp_holdings"]
    window = st.session_state["exp_window"]
    extra = holdings_tickers(holdings)
    model = build_model(window, extra)
    if "error" in model:
        st.error(model["error"])
        return
    good, bad, cash_d = split_holdings(holdings, model)
    if bad:
        st.warning(f"Left out (no price data, or less than {window} years of history): {', '.join(bad)}. "
                   "Try a shorter history window or check the ticker.")
    if len(good) < 2:
        st.warning("Enter at least 2 stocks (with enough price history) and a dollar amount above $0.")
        return

    tickers, _, w_cur, c_cur, total = current_weights(good, cash_d)
    now = port_metrics(w_cur, tickers, model)
    n = len(tickers)

    # ---------- current analysis
    st.markdown(f"#### Your portfolio today (${total:,.0f}, {n} stocks"
                + (f" + {c_cur:.0%} T-bills/cash)" if c_cur > 0 else ")"))
    metric_grid([
        ("Beta (CAPM risk)", f"{now['beta']:.2f}", "How much the portfolio moves with the market."),
        ("Expected return (FF)", f"{now['ff_er']:.1%}", "Fama-French estimate of the average yearly return."),
        ("Excess return", f"{now['excess']:.1%}", "Expected return minus the risk-free rate: the extra you expect for taking risk."),
        ("Volatility", f"{now['vol']:.1%}", "How much returns typically swing in a year."),
        ("Sharpe ratio", f"{now['sharpe']:.2f}", "Excess return ÷ volatility. Higher is better."),
        ("Past alpha", f"{now['alpha']:+.1%}", "Return beyond what CAPM predicted over the history window. Shown, not used to forecast."),
    ])
    with st.expander("See the CAPM, Fama-French, Sharpe and alpha details"):
        st.plotly_chart(sharpe_gauge(now["sharpe"], model["market_sharpe"], "Sharpe now"))
        sharpe_explanation(now, model)
        capm_box(now, model)
        ff_box(now, model)
        st.dataframe(model["stats"].loc[tickers, ["Company", "Beta (adj)", "Alpha", "FF Mkt", "FF SMB", "FF HML",
                                                 "FF E[R]", "Volatility", "Sharpe"]].round(3))
        alpha_explanation(model, window)

    # ---------- 2. goals (defaults = our recommendations)
    st.markdown("#### 2. Set your goals")
    st.write("Each slider starts at our recommendation. Move it to whatever you want; the explanation says why we suggest it.")
    allow_cash = st.session_state.get("exp_cash", True)   # read before the form so recommendations match it
    rec = recommended_settings(tuple(holdings), window, extra, allow_cash)
    sig = abs(hash((tuple(holdings), window, allow_cash)))   # sliders reset when the portfolio changes
    with st.form("goals_form"):
        st.checkbox("Allow T-bills (cash) in the recommendation", value=True, key="exp_cash")
        explain("T-bills have beta 0. Mixing them with your best stock mix can lower risk without lowering the "
                "Sharpe ratio (Capital Allocation Line). Turn off to stay 100% in stocks. Click **Update** after changing.")
        t_beta = st.slider("Target beta (CAPM market risk)", 0.3, 2.0, rec["beta"], 0.05, key=f"tb{sig}")
        explain(f"We recommend **{rec['beta']:.2f}**: your beta today, so we don't change how much market risk "
                "you take unless you ask. Lower = calmer portfolio; higher = bigger swings (and a higher CAPM return).")
        t_sharpe = st.slider("Target Sharpe ratio", 0.0, 1.5, rec["sharpe"], 0.01, key=f"ts{sig}")
        explain(f"Your Sharpe today is {now['sharpe']:.2f}. We recommend **{rec['sharpe']:.2f}**, the best your "
                "current stocks can reach **just by changing weights**"
                + (" (and adding T-bills)" if allow_cash else "") + " at your current beta. "
                "Set it higher and we'll suggest substitute stocks; set it lower and we'll make fewer changes.")
        t_smb = st.slider("Size tilt (Fama-French SMB)", -0.6, 1.0, rec["smb"], 0.05, key=f"tsmb{sig}")
        explain(f"Today: {now['smb']:+.2f} ({size_word(now['smb'])}). We recommend **{rec['smb']:+.2f}**, the size "
                "tilt of the highest-Sharpe version of your portfolio. Higher = more like small companies "
                "(historically higher return, more risk); lower = more like large ones.")
        t_hml = st.slider("Value tilt (Fama-French HML)", -0.8, 0.8, rec["hml"], 0.05, key=f"thml{sig}")
        explain(f"Today: {now['hml']:+.2f} ({value_word(now['hml'])}). We recommend **{rec['hml']:+.2f}** for the "
                "same reason. Higher = more value stocks (cheap vs. book value); lower = more growth stocks.")
        c1, c2 = st.columns(2)
        with c1:
            min_pct = st.slider("Minimum position (% of stocks)", 0.0, 20.0, rec["w_min"] * 100, 0.5, key=f"mn{sig}")
            explain(f"We recommend **{rec['w_min']:.1%}**, half an equal share (1/{n}), so every stock is big enough to matter.")
        with c2:
            max_pct = st.slider("Maximum position (% of stocks)", 5.0, 100.0, rec["w_max"] * 100, 0.5, key=f"mx{sig}")
            explain(f"We recommend **{rec['w_max']:.1%}**, double an equal share, so no single company can sink the portfolio.")
        max_swaps = st.slider("Max substitute stocks", 0, 5, 2, key=f"sw{sig}")
        explain("We recommend **2**: enough to fix most portfolios without replacing everything you own. "
                "We only substitute if changing weights alone can't reach your goals.")
        submitted = st.form_submit_button("Update recommendations", type="primary")
    if submitted:
        st.session_state["exp_seed"] = 0

    w_min, w_max = min_pct / 100, max_pct / 100
    if not limits_ok(n, w_min, w_max):
        st.error(f"With {n} stocks, the minimum × {n} must be ≤ 100% and the maximum × {n} must be ≥ 100%. "
                 "Adjust the position-size sliders.")
        return

    # ---------- 3. recommendations
    st.markdown("#### 3. Our recommendations")
    b1, b2 = st.columns(2)
    if b1.button("🔀 Shuffle substitutes", key="exp_shuffle"):
        st.session_state["exp_seed"] += 1
    if b2.button("↩️ Back to best substitutes", key="exp_reset"):
        st.session_state["exp_seed"] = 0
    seed = st.session_state["exp_seed"]

    targets = (t_beta, t_sharpe, t_smb, t_hml)
    res = recommend(tuple(holdings), window, extra, targets, w_min, w_max, max_swaps, seed, allow_cash)
    before, after = res["before"], res["after"]
    tdict = dict(zip(["beta", "sharpe", "smb", "hml"], targets))
    met = targets_met(after, tdict)

    summary = f"Sharpe ratio **{before['sharpe']:.2f} → {after['sharpe']:.2f}**, beta **{before['beta']:.2f} → {after['beta']:.2f}**."
    if res["swaps"]:
        summary += f" We changed weights and suggest **{len(res['swaps'])} substitute stock(s)**."
    else:
        summary += " We only changed weights." + (" No substitutes needed." if all(met.values()) else "")
    if res["c_final"] > res["c_cur"] + 0.005:
        summary += f" T-bills go from {res['c_cur']:.0%} to **{res['c_final']:.0%}**."
    st.success(summary)
    if seed > 0:
        st.info(f"Shuffle #{seed}: each substitute was drawn at random from the 5 best replacements, "
                "so you can see other options.")

    def yes(k):
        return "✅ Yes" if met[k] else "⚠️ Closest possible"
    comp = pd.DataFrame([
        ["Beta (CAPM)", f"{before['beta']:.2f}", f"{after['beta']:.2f}", f"{t_beta:.2f}", yes("beta")],
        ["Size tilt (SMB)", f"{before['smb']:+.2f}", f"{after['smb']:+.2f}", f"{t_smb:+.2f}", yes("smb")],
        ["Value tilt (HML)", f"{before['hml']:+.2f}", f"{after['hml']:+.2f}", f"{t_hml:+.2f}", yes("hml")],
        ["Sharpe ratio", f"{before['sharpe']:.2f}", f"{after['sharpe']:.2f}", f"≥ {t_sharpe:.2f}", yes("sharpe")],
        ["T-bills (cash)", f"{before['cash']:.1%}", f"{after['cash']:.1%}", "", ""],
        ["CAPM expected return", f"{before['capm_er']:.2%}", f"{after['capm_er']:.2%}", "", ""],
        ["FF expected return", f"{before['ff_er']:.2%}", f"{after['ff_er']:.2%}", "", ""],
        ["Excess return", f"{before['excess']:.2%}", f"{after['excess']:.2%}", "", ""],
        ["Volatility", f"{before['vol']:.2%}", f"{after['vol']:.2%}", "", ""],
        ["Past alpha", f"{before['alpha']:+.2%}", f"{after['alpha']:+.2%}", "", ""],
    ], columns=["Metric", "Now", "Recommended", "Your goal", "Reached?"])
    st.dataframe(comp, hide_index=True)
    if not all(met.values()):
        explain("'Closest possible' means the goal can't be fully reached with these stocks and position limits "
                "(or goals pull against each other). Try more substitutes, wider position limits, or a different goal.")

    st.plotly_chart(sharpe_gauge(after["sharpe"], model["market_sharpe"], "Recommended Sharpe",
                                 reference=before["sharpe"]))
    explain("The number under the gauge shows the change from your Sharpe ratio today.")

    # substitutions
    if res["swaps"]:
        st.markdown("**Substitute stocks**")
        s = model["stats"]
        for sw in res["swaps"]:
            o, i = s.loc[sw["out"]], s.loc[sw["in"]]
            alts = ", ".join(f"{a} ({company(a)})" for a in sw["alternatives"]) or "none"
            which = ("the lowest" if sw["rank"] == 1 else
                     f"the {'2nd' if sw['rank'] == 2 else '3rd'}-lowest (replacing the weaker ones didn't help)")
            st.markdown(
                f"- **Replace {sw['out']} ({o['Company']}) with {sw['in']} ({i['Company']}, {i['Sector']})**  \n"
                f"  Why: {sw['out']} has {which} expected return per unit of risk among your original stocks "
                f"(stock Sharpe {o['Sharpe']:.2f}). {sw['in']} (beta {i['Beta (adj)']:.2f}, "
                f"{size_word(i['FF SMB'])}, {value_word(i['FF HML'])}, stock Sharpe {i['Sharpe']:.2f}) "
                f"moved the portfolio closest to your goals.  \n  Other good options: {alts}")
    elif max_swaps == 0:
        explain("Substitutes are turned off (Max substitute stocks = 0).")

    # trades table + chart
    st.markdown("**What to buy and sell**")
    all_t = list(dict.fromkeys(res["original"] + res["final"]))
    show_cash = res["c_cur"] > 0 or res["c_final"] > 1e-6
    labels = all_t + ([CASH] if show_cash else [])
    cur_w = dict(zip(res["original"], res["w_cur"]))
    new_w = dict(zip(res["final"], res["w_final"]))
    if show_cash:
        cur_w[CASH], new_w[CASH] = res["c_cur"], res["c_final"]
    cur_d = dict(zip(labels, to_dollars([cur_w.get(t, 0.0) for t in labels], res["total"])))
    new_d = dict(zip(labels, to_dollars([new_w.get(t, 0.0) for t in labels], res["total"])))
    cur_p = dict(zip(labels, to_pct([cur_w.get(t, 0.0) for t in labels])))
    new_p = dict(zip(labels, to_pct([new_w.get(t, 0.0) for t in labels])))
    trades = pd.DataFrame([{
        "Holding": t, "Company": company(t),
        "Now $": cur_d[t], "Now %": cur_p[t], "Recommended %": new_p[t], "Recommended $": new_d[t],
        "Change $": round(new_d[t] - cur_d[t], 2),
        "Action": action_label(cur_d[t], new_d[t], res["total"]),
    } for t in labels])
    money = st.column_config.NumberColumn(format="$%.2f")
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(trades, hide_index=True, column_config={
        "Now $": money, "Recommended $": money, "Change $": money, "Now %": pct, "Recommended %": pct})
    st.plotly_chart(compare_bars(labels, cur_p, new_p))

    with st.expander("🔍 How did we come up with these recommendations?"):
        st.markdown(f"""
1. **Measure today's portfolio.** Your weights are your dollars ÷ ${res['total']:,.0f}. Beta, size, value and alpha are
   weighted averages of each holding's regression results (T-bills count as 0); volatility uses the covariance matrix.
2. **Change weights first.** An optimizer looks for the **smallest change to your weights** that gets beta, size,
   value and Sharpe as close as possible to your goals, keeping every stock between {w_min:.1%} and {w_max:.1%}
   of the stock part{' and up to ' + format(MAX_CASH, '.0%') + ' in T-bills' if allow_cash else ''}.
   Reweighting alone gives a Sharpe of **{res['reweight']['sharpe']:.2f}** and beta **{res['reweight']['beta']:.2f}**.
3. **Substitute only if needed.** If a goal is still missed, we replace your stock with the lowest *expected* stock
   Sharpe. We test **every stock in our universe** as its replacement and keep the one that gets closest to your goals,
   up to {max_swaps} substitute(s). **Shuffle** picks among the 5 best replacements instead.
4. **A goal is 'reached'** when beta is within ±{TOLERANCE['beta']}, size/value within ±{TOLERANCE['smb']},
   and the Sharpe ratio is at or above your target.
""")


# =============================================================================
# 12. APP LAYOUT
# =============================================================================
# "centered" keeps everything in one readable column on a laptop, and Streamlit stacks
# columns automatically on a phone.
st.set_page_config(page_title="Robo-Advisor", page_icon="📈", layout="centered")
st.title("Automated Financial Advisor")
st.caption("AC420 Project · Lydia Pelton & Sydney Pitts · Data: Yahoo Finance + Kenneth French Data Library")

tab_home, tab_start, tab_improve = st.tabs(["🏠 Home", "🌱 Start from scratch", "📈 Improve my portfolio"])
with tab_home:
    home_page()
with tab_start:
    beginner_tab()
with tab_improve:
    existing_tab()
