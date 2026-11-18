"""
AC420 Robo-Advisor  —  Lydia Pelton & Sydney Pitts
=================================================
Run with:   streamlit run app.py

HOW THE FILE IS ORGANIZED (read top to bottom)
  1. Assumptions ............ every fixed number the model uses, with its source
  2. Stock universe ......... the stocks the advisor is allowed to pick from
  3. Data ................... prices (Yahoo Finance), Fama-French factors (Ken French library)
  4. Models ................. CAPM + Fama-French regressions, expected returns, risk
  5. Portfolio math ......... beta, factor loadings, return, volatility, Sharpe of a portfolio
  6. Tab 1 logic ............ age -> risk range -> pick stocks -> max-Sharpe weights
  7. Tab 2 logic ............ analyze holdings -> recommended targets -> reweight + substitutes
  8. Display helpers ........ gauge, charts, plain-English explanations
  9. Tab 1 page ............. "Start from scratch"
 10. Tab 2 page ............. "Improve my portfolio"
 11. Methods page section ... how it works + assumptions, shown in the app
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
    return UNIVERSE.get(t, (t, "Your stock", "—"))[0]


def sector(t):
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
       CAPM regression:  R - RF = a + b*(Mkt-RF)                    -> beta
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
    """Every portfolio number is a weighted average of the stocks' numbers,
    except volatility, which uses the covariance matrix: sqrt(w' Cov w)."""
    w = np.asarray(w, dtype=float)
    s = model["stats"].loc[list(tickers)]
    cov = model["cov"].loc[list(tickers), list(tickers)].values
    rf = model["rf"]
    vol = float(np.sqrt(w @ cov @ w))
    ff_er = float(w @ s["FF E[R]"].values)
    return {
        "beta": float(w @ s["Beta (adj)"].values),
        "mkt": float(w @ s["FF Mkt"].values),
        "smb": float(w @ s["FF SMB"].values),
        "hml": float(w @ s["FF HML"].values),
        "capm_er": float(w @ s["CAPM E[R]"].values),
        "ff_er": ff_er,
        "excess": ff_er - rf,
        "vol": vol,
        "sharpe": (ff_er - rf) / vol,
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
# 6. TAB 1 LOGIC: AGE -> RISK RANGE -> PICK STOCKS -> MAX-SHARPE WEIGHTS
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
    """Weights that MAXIMIZE the Sharpe ratio subject to:
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


# =============================================================================
# 7. TAB 2 LOGIC: ANALYZE -> RECOMMENDED TARGETS -> REWEIGHT + SUBSTITUTES
# =============================================================================

def holdings_tickers(holdings):
    return tuple(sorted(t for t, _ in holdings))


def valid_holdings(holdings, model):
    stats = model["stats"]
    good = [(t, d) for t, d in holdings if t in stats.index]
    bad = [t for t, _ in holdings if t not in stats.index]
    return good, bad


@st.cache_data(show_spinner=False)
def recommended_settings(holdings, window, extra):
    """Default slider values for Tab 2, each with a reason shown in the app.
    extra = every ticker the user typed (same model as the page, so no re-download)."""
    model = build_model(window, extra)
    good, _ = valid_holdings(holdings, model)
    tickers = [t for t, _ in good]
    dollars = np.array([d for _, d in good])
    w_cur = dollars / dollars.sum()
    now = port_metrics(w_cur, tickers, model)

    w_min, w_max = recommended_limits(len(tickers))
    beta = round(float(np.clip(now["beta"], 0.3, 2.0)), 2)
    # "Best version of your portfolio": the highest-Sharpe weights for the SAME stocks at
    # your current beta. Its Sharpe, size and value become the recommended goals.
    w_best, _, _ = max_sharpe(tickers, model, beta, w_min, w_max)
    best = port_metrics(w_best, tickers, model)
    smb = round(float(np.clip(best["smb"], -0.6, 1.0)), 2)    # kept inside the slider ranges
    hml = round(float(np.clip(best["hml"], -0.8, 0.8)), 2)
    # check what the Tab 2 optimizer actually reaches with these goals, and recommend
    # that Sharpe (rounded down) so the default goals are always reachable
    goals = {"beta": beta, "smb": smb, "hml": hml, "sharpe": best["sharpe"]}
    w_soft, _ = soft_optimize(tickers, w_cur, goals, model, w_min, w_max)
    reachable = port_metrics(w_soft, tickers, model)["sharpe"]
    sharpe = np.floor(reachable * 100) / 100
    return {
        "beta": beta, "smb": smb, "hml": hml,
        "sharpe": float(np.clip(sharpe, 0.0, 1.5)),
        "w_min": w_min, "w_max": w_max,
    }


def targets_met(m, targets):
    return {
        "beta": abs(m["beta"] - targets["beta"]) <= TOLERANCE["beta"],
        "smb": abs(m["smb"] - targets["smb"]) <= TOLERANCE["smb"],
        "hml": abs(m["hml"] - targets["hml"]) <= TOLERANCE["hml"],
        "sharpe": m["sharpe"] >= targets["sharpe"] - TOLERANCE["sharpe"],
    }


def soft_optimize(tickers, base_w, targets, model, w_min, w_max):
    """Find weights that get as close as possible to ALL goals at once, with the
    smallest change from base_w (what you own now). Minimizes:
        50*(beta - target)^2 + 20*(size - target)^2 + 20*(value - target)^2
      + 50*(Sharpe shortfall below target)^2 + 1*(sum of squared weight changes)
    Goals are 'soft', so the optimizer always returns an answer even when a goal
    can't be fully reached; the app then reports how close it got."""
    b, smb, hml, mu, cov = _arrays(tickers, model)
    rf, n = model["rf"], len(tickers)

    def objective(w):
        sr = (w @ mu - rf) / np.sqrt(w @ cov @ w)
        return (W_BETA * (w @ b - targets["beta"]) ** 2
                + W_TILT * (w @ smb - targets["smb"]) ** 2
                + W_TILT * (w @ hml - targets["hml"]) ** 2
                + W_SHARPE * max(0.0, targets["sharpe"] - sr) ** 2
                + W_TURN * np.sum((w - base_w) ** 2))

    start = np.clip(base_w, w_min, w_max)
    start = start / start.sum()
    res = minimize(objective, start, method="SLSQP", bounds=[(w_min, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 300})
    w = res.x if abs(res.x.sum() - 1) < 1e-4 else start
    w = np.clip(w, w_min, w_max)
    w = w / w.sum()
    return w, float(objective(w))


@st.cache_data(show_spinner="Finding the best adjustments to your portfolio...")
def recommend(holdings, window, extra, targets_tuple, w_min, w_max, max_swaps, seed):
    """Step 1: reweight the stocks you already own (soft_optimize).
    Step 2: if goals still aren't met, swap out your holding with the lowest stock Sharpe.
            Every stock in the universe is tried as its replacement (weights re-optimized
            each time); the one that gets closest to your goals wins. If no replacement
            helps, the next-weakest holding is tried (up to 3).
            Shuffle (seed > 0): choose at random among the 5 best replacements instead.
    Repeats step 2 up to max_swaps times, stopping early once goals are met."""
    model = build_model(window, extra)
    stats = model["stats"]
    good, _ = valid_holdings(holdings, model)
    original = [t for t, _ in good]
    dollars = np.array([d for _, d in good])
    w_cur = dollars / dollars.sum()
    targets = dict(zip(["beta", "sharpe", "smb", "hml"], targets_tuple))

    current, base = list(original), w_cur.copy()
    w, score = soft_optimize(current, base, targets, model, w_min, w_max)
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
            for c in stats.index:
                if c in current or c in removed:
                    continue
                cand = list(current)
                cand[i] = c
                wn, sn = soft_optimize(cand, new_base, targets, model, w_min, w_max)
                trials.append((sn + sell_penalty, c, wn))
            improving = sorted([t for t in trials if t[0] < score - 1e-4], key=lambda x: x[0])
            if improving:
                break
        if not improving:
            break

        k = 0 if seed == 0 else int(rng.integers(min(5, len(improving))))
        new_score, c, wn = improving[k]
        swaps.append({"out": weakest, "in": c, "rank": rank,
                      "alternatives": [t[1] for t in improving[:6] if t[1] != c][:3]})
        removed.add(weakest)
        current[i] = c
        base = new_base
        w, score = wn, new_score

    return {
        "original": original, "w_cur": w_cur, "total": float(dollars.sum()),
        "dollars_cur": dollars, "final": current, "w_final": w, "swaps": swaps,
        "before": port_metrics(w_cur, original, model),
        "reweight": reweight_metrics,
        "after": port_metrics(w, current, model),
    }


# =============================================================================
# 8. DISPLAY HELPERS
# =============================================================================

def explain(text):
    st.caption(f"💡 {text}")


def size_word(smb):
    return "small-company" if smb > 0.25 else ("large-company" if smb < 0 else "mid-size")


def value_word(hml):
    return "value" if hml > 0.2 else ("growth" if hml < -0.2 else "blend")


def sharpe_band(x):
    for lo, hi, label in SHARPE_BANDS:
        if lo <= x < hi:
            return label
    return "Low"


def sharpe_gauge(value, market_sharpe, title="Sharpe ratio"):
    top = max(1.2, value * 1.15)
    colors = ["#f4cccc", "#fff2cc", "#d9ead3", "#b6d7a8"]
    steps = [{"range": [lo, min(hi, top)], "color": c} for (lo, hi, _), c in zip(SHARPE_BANDS, colors)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"valueformat": ".2f"},
        title={"text": f"{title}<br><span style='font-size:0.8em'>{sharpe_band(value)}</span>"},
        gauge={"axis": {"range": [0, top]}, "bar": {"color": "#1f4e79"}, "steps": steps,
               "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.85,
                             "value": market_sharpe}}))
    fig.update_layout(height=260, margin=dict(t=70, b=10, l=30, r=30))
    return fig


def sharpe_explanation(m, model):
    st.markdown(
        f"**Sharpe ratio = (expected return − risk-free rate) ÷ volatility** "
        f"= ({m['ff_er']:.2%} − {model['rf']:.2%}) ÷ {m['vol']:.2%} = **{m['sharpe']:.2f}**")
    explain("It measures how much extra return you expect for each unit of risk. Higher is better. "
            f"The black line on the gauge is the overall stock market ({model['market_sharpe']:.2f}) "
            "using the same assumptions. The color bands are our rule-of-thumb labels.")


def return_waterfall(m, model):
    """Fama-French expected return, split into where it comes from."""
    p, rf = model["prem"], model["rf"]
    parts = [("Risk-free rate", rf), ("Market factor", m["mkt"] * p["MKT"]),
             ("Size factor (SMB)", m["smb"] * p["SMB"]), ("Value factor (HML)", m["hml"] * p["HML"])]
    fig = go.Figure(go.Waterfall(
        x=[x for x, _ in parts] + ["Expected return"],
        measure=["relative"] * len(parts) + ["total"],
        y=[v * 100 for _, v in parts] + [0],
        text=[f"{v:+.2%}" for _, v in parts] + [f"{m['ff_er']:.2%}"],
        textposition="outside"))
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
                "CAPM says the only risk you get paid for is market risk, so return rises in a straight line with beta.")


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
                "their accounting value), negative toward 'growth' stocks. "
                f"Value has historically earned {p['HML']:.2%} a year more than growth.")


def weights_pie(tickers, weights, title=None):
    df = pd.DataFrame({"Ticker": tickers, "Company": [company(t) for t in tickers],
                       "Weight %": to_pct(weights)})
    fig = px.pie(df, names="Ticker", values="Weight %", hover_data=["Company"], title=title)
    fig.update_traces(textposition="inside", textinfo="label+percent")
    fig.update_layout(height=380, margin=dict(t=40 if title else 10, b=10, l=10, r=10))
    return fig


# =============================================================================
# 9. TAB 1 PAGE: START FROM SCRATCH
# =============================================================================

def beginner_tab():
    st.subheader("Build your first portfolio")
    st.write("Answer four questions and we'll build a diversified stock portfolio for you. "
             "Every number comes with a short explanation of what it means.")

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
                    "Younger investors have more years to recover from a market drop, so their range is higher.")
            risk = st.slider("Risk tolerance (within your age range)", 0, 100, 50, key="beg_risk")
            target = target_beta_from_slider(age, risk)
            explain(f"0 = the safest end of your range, 100 = the riskiest. Your **target beta is {target:.2f}** "
                    "(beta measures how much a portfolio moves with the stock market; 1.0 = the same as the market).")
        with c2:
            amount = st.number_input("Amount to invest ($)", min_value=100, value=10_000, step=500, key="beg_amount")
            explain("We split this amount across your stocks.")
            n = st.slider("Number of stocks", 10, 30, 15, key="beg_n")
            explain("More stocks = more diversification, so one company having a bad year hurts you less.")

    # ---------- shuffle controls
    st.session_state.setdefault("beg_seed", 0)
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("🔀 Shuffle stocks", key="beg_shuffle"):
        st.session_state["beg_seed"] += 1
    if b2.button("↩️ Back to best picks", key="beg_reset"):
        st.session_state["beg_seed"] = 0
    seed = st.session_state["beg_seed"]

    # ---------- model
    picks, pool = pick_beginner_stocks(model, target, n, seed)
    w_min, w_max = recommended_limits(n)
    w, tb_used, _ = max_sharpe(picks, model, target, w_min, w_max)
    m = port_metrics(w, picks, model)

    if seed > 0:
        st.info(f"Shuffle #{seed}: these {n} stocks were drawn at random from the {len(pool)} stocks closest "
                "to your target beta. Shuffled portfolios usually score a little lower on the Sharpe gauge "
                "than the best picks. Compare them!")
    if abs(tb_used - target) > 1e-3:
        st.info(f"A beta of {target:.2f} isn't reachable with these stocks, so we used the closest possible "
                f"({tb_used:.2f}).")

    # ---------- headline numbers
    st.markdown("### Your portfolio")
    c = st.columns(4)
    c[0].metric("Risk (beta)", f"{m['beta']:.2f}", help="How much the portfolio moves with the market. 1.0 = same as the market.")
    c[1].metric("Expected return", f"{m['ff_er']:.1%}", help="Fama-French estimate of the average yearly return.")
    c[2].metric("Volatility", f"{m['vol']:.1%}", help="How much returns typically swing in a year (standard deviation). In a typical year, returns land within about ± this amount of the average.")
    c[3].metric("Sharpe ratio", f"{m['sharpe']:.2f}", help="Return per unit of risk. Higher is better.")

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(sharpe_gauge(m["sharpe"], model["market_sharpe"]))
        sharpe_explanation(m, model)
    with right:
        st.plotly_chart(weights_pie(picks, w))

    # ---------- holdings table with reasons
    st.markdown("### Your stocks and why we chose them")
    s = model["stats"]
    pool_rank = s.loc[pool, "Sharpe"].rank(ascending=False).astype(int)
    dollars = to_dollars(w, amount)
    pcts = to_pct(w)
    rows = []
    for t, wt, d in zip(picks, pcts, dollars):
        r = s.loc[t]
        if seed == 0:
            why = (f"Beta {r['Beta (adj)']:.2f} is close to your target {tb_used:.2f}; ranked #{pool_rank[t]} of "
                   f"{len(pool)} candidates on return per unit of risk (stock Sharpe {r['Sharpe']:.2f}).")
        else:
            why = (f"Beta {r['Beta (adj)']:.2f} is close to your target {tb_used:.2f}; drawn at random from the "
                   f"{len(pool)} closest-beta stocks.")
        rows.append({"Ticker": t, "Company": r["Company"], "Sector": r["Sector"],
                     "Beta": round(r["Beta (adj)"], 2),
                     "Style": f"{size_word(r['FF SMB'])}, {value_word(r['FF HML'])}",
                     "Exp. return": round(r["FF E[R]"] * 100, 1),
                     "Weight %": wt, "$ Amount": d, "Why chosen": why})
    table = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
    st.dataframe(table, hide_index=True, column_config={
        "Exp. return": st.column_config.NumberColumn(format="%.1f%%"),
        "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
        "$ Amount": st.column_config.NumberColumn(format="$%.2f")})
    st.write(f"**Total:** {pcts.sum():.1f}% · ${dollars.sum():,.2f}")

    # ---------- CAPM and Fama-French side by side
    st.markdown("### What the models say")
    c1, c2 = st.columns(2)
    with c1:
        capm_box(m, model)
    with c2:
        ff_box(m, model)
    explain("We show both models. The Fama-French model is the one we use to pick stocks and set weights, "
            "because it adds size and value to the market factor and explains returns better than CAPM alone.")

    # ---------- how the stocks were picked
    with st.expander("🔍 How did the model pick these stocks?"):
        st.markdown(f"""
1. **Universe:** we start from {len(s)} U.S. stocks across every sector, both large and small/mid-size companies (full list below).
2. **Your target beta:** your age range ({floor:.2f}–{ceiling:.2f}) and risk slider ({risk}) give a target beta of **{target:.2f}**.
3. **Candidate pool:** we keep the **{len(pool)} stocks** (2 × the {n} you asked for) whose beta is closest to {target:.2f}.
4. **Pick {n}:** from the pool we keep the {n} with the highest *stock Sharpe ratio*: Fama-French expected return above the risk-free rate, divided by the stock's own volatility. **Shuffle** instead draws {n} at random from the pool.
5. **Weights:** an optimizer chooses the weights that give the **highest portfolio Sharpe ratio** while keeping portfolio beta at {tb_used:.2f} and each stock between **{w_min:.1%} and {w_max:.1%}** (half to double an equal share, so no single stock dominates).
""")
        pool_df = s.loc[pool, ["Company", "Sector", "Beta (adj)", "FF E[R]", "Volatility", "Sharpe"]].copy()
        pool_df.insert(0, "Selected", ["✅" if t in picks else "" for t in pool_df.index])
        st.dataframe(pool_df.sort_values("Sharpe", ascending=False).round(3))


# =============================================================================
# 10. TAB 2 PAGE: IMPROVE MY PORTFOLIO
# =============================================================================

def clean_holdings(df):
    d = df.copy()
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
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
    st.write("Enter your stocks, see how your portfolio scores on CAPM, Fama-French and the Sharpe ratio, "
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
            column_config={"Ticker": st.column_config.TextColumn(help="Stock symbol, e.g. AAPL"),
                           "$ Amount": st.column_config.NumberColumn(min_value=0, format="$%.2f")})
        explain("Type each stock's ticker and how many dollars you have in it. Add rows with the + at the bottom. "
                "Any U.S. stock on Yahoo Finance works, not just the ones in our list.")
        window = st.select_slider("History window", options=WINDOW_OPTIONS, value=DEFAULT_WINDOW_YEARS,
                                  format_func=lambda y: f"{y} years")
        explain("How many years of monthly returns we use to estimate beta and the Fama-French loadings. "
                "**We recommend 5 years** (60 months), the most common choice for beta estimates. "
                "3 years reacts faster to recent changes but is noisier; 10 years is more stable but includes "
                "older data that may no longer describe the company.")
        if st.form_submit_button("Analyze my portfolio", type="primary"):
            st.session_state["exp_holdings"] = clean_holdings(edited)
            st.session_state["exp_window"] = window
            st.session_state["exp_seed"] = 0

    holdings = st.session_state["exp_holdings"]
    window = st.session_state["exp_window"]
    if len(holdings) < 2:
        st.warning("Enter at least 2 stocks with a dollar amount above $0.")
        return

    extra = holdings_tickers(holdings)
    model = build_model(window, extra)
    if "error" in model:
        st.error(model["error"])
        return
    good, bad = valid_holdings(holdings, model)
    if bad:
        st.warning(f"Left out (no price data, or less than {window} years of history): {', '.join(bad)}. "
                   "Try a shorter history window or check the ticker.")
    if len(good) < 2:
        st.warning("We need at least 2 stocks with enough history to analyze.")
        return

    tickers = [t for t, _ in good]
    dollars = np.array([d for _, d in good])
    total = dollars.sum()
    w_cur = dollars / total
    now = port_metrics(w_cur, tickers, model)

    # ---------- current analysis
    st.markdown(f"#### Your portfolio today (${total:,.0f}, {len(tickers)} stocks)")
    c = st.columns(5)
    c[0].metric("Beta (CAPM risk)", f"{now['beta']:.2f}")
    c[1].metric("Expected return (FF)", f"{now['ff_er']:.1%}")
    c[2].metric("Excess return", f"{now['excess']:.1%}", help="Expected return minus the risk-free rate: the extra you expect for taking risk.")
    c[3].metric("Volatility", f"{now['vol']:.1%}")
    c[4].metric("Sharpe ratio", f"{now['sharpe']:.2f}")
    with st.expander("See the CAPM, Fama-French and Sharpe details for your current portfolio"):
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(sharpe_gauge(now["sharpe"], model["market_sharpe"], "Current Sharpe"))
            sharpe_explanation(now, model)
        with g2:
            st.plotly_chart(weights_pie(tickers, w_cur, "Current weights"))
        f1, f2 = st.columns(2)
        with f1:
            capm_box(now, model)
        with f2:
            ff_box(now, model)
        st.dataframe(model["stats"].loc[tickers, ["Company", "Beta (adj)", "FF Mkt", "FF SMB", "FF HML",
                                                 "FF E[R]", "Volatility", "Sharpe"]].round(3))

    # ---------- 2. goals (defaults = our recommendations)
    rec = recommended_settings(tuple(good), window, extra)
    sig = abs(hash((tuple(good), window)))   # sliders reset to new recommendations when the portfolio changes
    n = len(tickers)
    st.markdown("#### 2. Set your goals")
    st.write("Each slider starts at our recommendation. Move it to whatever you want; the explanation says why we suggest it.")
    with st.form("goals_form"):
        g1, g2 = st.columns(2)
        with g1:
            t_beta = st.slider("Target beta (CAPM market risk)", 0.3, 2.0, rec["beta"], 0.05, key=f"tb{sig}")
            explain(f"Based on your portfolio we recommend **{rec['beta']:.2f}**, because that's your beta today, "
                    "so we don't change how much market risk you take unless you ask. Lower it for a calmer "
                    "portfolio, raise it for bigger swings (and a higher CAPM expected return).")
            t_sharpe = st.slider("Target Sharpe ratio", 0.0, 1.5, rec["sharpe"], 0.01, key=f"ts{sig}")
            explain(f"Your Sharpe today is {now['sharpe']:.2f}. We recommend **{rec['sharpe']:.2f}**, the best your "
                    "current stocks can reach **just by changing weights** at your current beta. "
                    "Set it higher and we'll suggest substitute stocks to get there; set it lower and we'll make fewer changes.")
        with g2:
            t_smb = st.slider("Size tilt (Fama-French SMB)", -0.6, 1.0, rec["smb"], 0.05, key=f"tsmb{sig}")
            explain(f"Today: {now['smb']:+.2f} ({size_word(now['smb'])}). We recommend **{rec['smb']:+.2f}**, the size "
                    "tilt of the highest-Sharpe version of your current portfolio. Higher = more like small companies "
                    "(historically higher return, more risk); lower = more like large ones. Very strong tilts in "
                    "either direction add risk that isn't market risk.")
            t_hml = st.slider("Value tilt (Fama-French HML)", -0.8, 0.8, rec["hml"], 0.05, key=f"thml{sig}")
            explain(f"Today: {now['hml']:+.2f} ({value_word(now['hml'])}). We recommend **{rec['hml']:+.2f}** for the "
                    "same reason. Higher = more value stocks (cheap vs. book value); lower = more growth stocks.")
        p1, p2, p3 = st.columns(3)
        with p1:
            min_pct = st.slider("Minimum position size (%)", 0.0, 20.0, rec["w_min"] * 100, 0.5, key=f"mn{sig}")
            explain(f"We recommend **{rec['w_min']:.1%}**, half of an equal share (1/{n}), so every stock "
                    "is big enough to matter.")
        with p2:
            max_pct = st.slider("Maximum position size (%)", 5.0, 100.0, rec["w_max"] * 100, 0.5, key=f"mx{sig}")
            explain(f"We recommend **{rec['w_max']:.1%}**, double an equal share, so no single company "
                    "can sink the portfolio.")
        with p3:
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
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("🔀 Shuffle substitutes", key="exp_shuffle"):
        st.session_state["exp_seed"] += 1
    if b2.button("↩️ Back to best substitutes", key="exp_reset"):
        st.session_state["exp_seed"] = 0
    seed = st.session_state["exp_seed"]

    targets = (t_beta, t_sharpe, t_smb, t_hml)
    res = recommend(tuple(good), window, extra, targets, w_min, w_max, max_swaps, seed)
    before, after = res["before"], res["after"]
    tdict = dict(zip(["beta", "sharpe", "smb", "hml"], targets))
    met = targets_met(after, tdict)

    summary = f"Sharpe ratio **{before['sharpe']:.2f} → {after['sharpe']:.2f}**, beta **{before['beta']:.2f} → {after['beta']:.2f}**."
    if res["swaps"]:
        summary += f" We changed weights and suggest **{len(res['swaps'])} substitute stock(s)**."
    else:
        summary += " We only changed weights. No substitutes needed." if all(met.values()) else " We only changed weights."
    st.success(summary)
    if seed > 0:
        st.info(f"Shuffle #{seed}: each substitute was drawn at random from the 5 best replacements, "
                "so you can see other options.")

    # before / after / target table
    def yes(k):
        return "✅ Yes" if met[k] else "⚠️ Closest possible"
    comp = pd.DataFrame([
        ["Beta (CAPM)", f"{before['beta']:.2f}", f"{after['beta']:.2f}", f"{t_beta:.2f}", yes("beta")],
        ["Size tilt (SMB)", f"{before['smb']:+.2f}", f"{after['smb']:+.2f}", f"{t_smb:+.2f}", yes("smb")],
        ["Value tilt (HML)", f"{before['hml']:+.2f}", f"{after['hml']:+.2f}", f"{t_hml:+.2f}", yes("hml")],
        ["Sharpe ratio", f"{before['sharpe']:.2f}", f"{after['sharpe']:.2f}", f"≥ {t_sharpe:.2f}", yes("sharpe")],
        ["CAPM expected return", f"{before['capm_er']:.2%}", f"{after['capm_er']:.2%}", "", ""],
        ["FF expected return", f"{before['ff_er']:.2%}", f"{after['ff_er']:.2%}", "", ""],
        ["Excess return", f"{before['excess']:.2%}", f"{after['excess']:.2%}", "", ""],
        ["Volatility", f"{before['vol']:.2%}", f"{after['vol']:.2%}", "", ""],
    ], columns=["Metric", "Now", "Recommended", "Your goal", "Goal reached?"])
    st.dataframe(comp, hide_index=True)
    if not all(met.values()):
        explain("'Closest possible' means the goal can't be fully reached with these stocks and position limits "
                "(or goals pull against each other). Try more substitutes, wider position limits, or a different goal.")

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

    # trades table
    st.markdown("**What to buy and sell**")
    all_t = list(dict.fromkeys(res["original"] + res["final"]))
    cur_d = dict(zip(res["original"], res["dollars_cur"]))
    new_d = dict(zip(res["final"], to_dollars(res["w_final"], res["total"])))
    cur_p = dict(zip(res["original"], to_pct(res["w_cur"])))
    new_p = dict(zip(res["final"], to_pct(res["w_final"])))
    trades = pd.DataFrame([{
        "Ticker": t, "Company": company(t),
        "Now $": cur_d.get(t, 0.0), "Now %": cur_p.get(t, 0.0),
        "Recommended %": new_p.get(t, 0.0),
        "Recommended $": new_d.get(t, 0.0),
        "Change $": round(new_d.get(t, 0.0) - cur_d.get(t, 0.0), 2),
        "Action": action_label(cur_d.get(t, 0.0), new_d.get(t, 0.0), res["total"]),
    } for t in all_t])
    money = st.column_config.NumberColumn(format="$%.2f")
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(trades, hide_index=True, column_config={
        "Now $": money, "Recommended $": money, "Change $": money, "Now %": pct, "Recommended %": pct})

    c1, c2 = st.columns(2)
    c1.plotly_chart(weights_pie(res["original"], res["w_cur"], "Now"))
    c2.plotly_chart(weights_pie(res["final"], res["w_final"], "Recommended"))
    g1, g2 = st.columns(2)
    g1.plotly_chart(sharpe_gauge(before["sharpe"], model["market_sharpe"], "Sharpe now"))
    g2.plotly_chart(sharpe_gauge(after["sharpe"], model["market_sharpe"], "Sharpe recommended"))

    with st.expander("🔍 How did we come up with these recommendations?"):
        st.markdown(f"""
1. **Measure today's portfolio.** Your weights are your dollars ÷ ${res['total']:,.0f}. Beta, size and value are
   weighted averages of each stock's regression results; volatility uses the covariance matrix.
2. **Change weights first.** An optimizer looks for the **smallest change to your weights** that gets beta, size,
   value and Sharpe as close as possible to your goals, keeping every stock between {w_min:.1%} and {w_max:.1%}.
   Reweighting alone gives a Sharpe of **{res['reweight']['sharpe']:.2f}** and beta **{res['reweight']['beta']:.2f}**.
3. **Substitute only if needed.** If a goal is still missed, we replace your stock with the lowest stock Sharpe
   (expected return per unit of risk). We test **every stock in our universe** as its replacement and keep the one
   that gets closest to your goals, up to {max_swaps} substitute(s). **Shuffle** picks among the 5 best replacements instead.
4. **A goal is 'reached'** when beta is within ±{TOLERANCE['beta']}, size/value within ±{TOLERANCE['smb']},
   and the Sharpe ratio is at or above your target.
""")


# =============================================================================
# 11. METHODS & ASSUMPTIONS (shown at the bottom of the app)
# =============================================================================

def methods_section():
    model = build_model(DEFAULT_WINDOW_YEARS)
    if "error" in model:
        return
    p = model["prem"]
    with st.expander("📘 How this app works: data, models and assumptions"):
        st.markdown(f"""
**Data sources**
| What | Source |
|---|---|
| Stock prices | Yahoo Finance via `yfinance`: monthly adjusted closes (include dividends and splits) |
| Fama-French factors (Mkt-RF, SMB, HML, RF) | {model['ff_source']}, the same data WRDS distributes |
| Current risk-free rate | {model['rf_source']}: **{model['rf']:.2%}** |
| Estimation window (Tab 1) | {model['start']} to {model['end']} ({model['n_months']} months). Tab 2 lets you choose 3, 5 or 10 years. |

**Models (linear regressions, one per stock, on monthly excess returns)**
- **CAPM:** R − RF = α + β(Mkt−RF). β is then **Blume-adjusted**: 0.67 × β + 0.33, because betas drift toward 1 over time (Blume, 1971).
- **Fama-French 3-factor:** R − RF = α + b(Mkt−RF) + s·SMB + h·HML.
- **Expected returns:** CAPM: RF + β × market premium. Fama-French: RF + b × market premium + s × SMB premium + h × HML premium. We assume α = 0 going forward, since past α rarely repeats.
- **Factor premiums** = average annual factor return since July 1963 (the start of Fama & French's 1993 sample):
  market **{p['MKT']:.2%}**, SMB **{p['SMB']:.2%}**, HML **{p['HML']:.2%}**.
- **Risk:** covariance of monthly returns with **Ledoit-Wolf shrinkage** (a standard way to reduce noise when there are many stocks and few months), × 12 to annualize. Volatility = √(w′Σw).
- **Sharpe ratio** = (Fama-French expected return − RF) ÷ volatility. Market Sharpe for comparison: **{model['market_sharpe']:.2f}**.

**Tab 1 rules (our assumptions)**
| Age | Beta range |
|---|---|
| under 30 | 0.90 – 1.50 |
| 30–44 | 0.80 – 1.30 |
| 45–59 | 0.60 – 1.10 |
| 60+ | 0.40 – 0.90 |

Based on the idea behind the "110 − age = % in stocks" rule and target-date funds: more years until you need the money = more risk you can recover from. The risk slider picks a point in the range. Stocks: the 2N closest in beta, then the N best stock Sharpes. Weights: maximum Sharpe with beta fixed at the target, each stock between ½ and 2× an equal share.

**Tab 2 rules**
Recommended goals come from the **highest-Sharpe version of your current portfolio** (same stocks, new weights, same beta as today): beta = today's beta; Sharpe, size and value = that version's values; position limits = ½ and 2× an equal share; 2 substitutes.
Optimizer: minimizes 50·(beta gap)² + 20·(size gap)² + 20·(value gap)² + 50·(Sharpe shortfall)² + 1·(weight changes)².

**Limitations:** estimates come from past data and may not hold in the future. The universe is {len(model['stats'])} stocks. Taxes and trading costs are ignored. The Fama-French file is published with a 1–2 month delay. This is a class project, not investment advice.
""")
        if model["missing"]:
            st.caption("Stocks in our list skipped for missing data: " + ", ".join(model["missing"]))


# =============================================================================
# APP LAYOUT
# =============================================================================
st.set_page_config(page_title="Robo-Advisor", page_icon="📈", layout="wide")
st.title("Automated Financial Advisor")
st.caption("AC420 Project · Lydia Pelton & Sydney Pitts · Data: Yahoo Finance + Kenneth French Data Library")

tab1, tab2 = st.tabs(["🌱 Start from scratch", "📈 Improve my portfolio"])
with tab1:
    beginner_tab()
with tab2:
    existing_tab()

st.divider()
methods_section()
