import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, linprog

# ---------------- Assumptions ----------------
MRP = 0.055            # market risk premium (annual)
MARKET = "SPY"         # market proxy
MIN_WEIGHT = 0.01      # smallest position allowed
MAX_WEIGHT = 0.20      # largest position allowed

UNIVERSE = {
    # Utilities / staples / telecom (usually low beta)
    "DUK": "Duke Energy", "SO": "Southern Co", "NEE": "NextEra Energy", "AEP": "American Electric Power",
    "ED": "Consolidated Edison", "XEL": "Xcel Energy", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "PG": "Procter & Gamble", "CL": "Colgate-Palmolive", "GIS": "General Mills", "KMB": "Kimberly-Clark",
    "WMT": "Walmart", "COST": "Costco", "MCD": "McDonald's", "VZ": "Verizon", "T": "AT&T",
    # Health care
    "JNJ": "Johnson & Johnson", "MRK": "Merck", "PFE": "Pfizer", "ABBV": "AbbVie", "UNH": "UnitedHealth",
    "LLY": "Eli Lilly", "TMO": "Thermo Fisher", "ABT": "Abbott Labs",
    # Financials
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs", "MS": "Morgan Stanley",
    "V": "Visa", "MA": "Mastercard", "BRK-B": "Berkshire Hathaway", "AXP": "American Express",
    # Industrials / energy / materials
    "CAT": "Caterpillar", "DE": "Deere", "HON": "Honeywell", "UPS": "UPS", "GE": "GE Aerospace",
    "LMT": "Lockheed Martin", "XOM": "ExxonMobil", "CVX": "Chevron", "LIN": "Linde",
    # Consumer discretionary
    "HD": "Home Depot", "NKE": "Nike", "SBUX": "Starbucks", "AMZN": "Amazon", "TSLA": "Tesla",
    # Tech / communication (usually high beta)
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "META": "Meta Platforms",
    "NVDA": "NVIDIA", "AMD": "AMD", "AVGO": "Broadcom", "ORCL": "Oracle", "CRM": "Salesforce",
    "ADBE": "Adobe", "INTC": "Intel", "CSCO": "Cisco", "NFLX": "Netflix",
}


# ---------------- Data (cached for 1 hour) ----------------
@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance...")
def load_data():
    tickers = list(UNIVERSE.keys())
    prices = yf.download(tickers + [MARKET], period="5y", interval="1mo",
                         auto_adjust=True, progress=False)["Close"]
    # drop stocks missing more than 10% of months
    prices = prices.dropna(axis=1, thresh=int(0.9 * len(prices)))

    irx = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)["Close"]
    rf = float(irx.dropna().iloc[-1].squeeze()) / 100
    return prices, rf


# ---------------- Linear models: beta + CAPM ----------------
@st.cache_data(ttl=3600)
def estimate_stats(prices, rf):
    returns = prices.pct_change().dropna(how="all")
    excess = returns - rf / 12

    rows = []
    for t in excess.columns:
        if t == MARKET:
            continue
        data = excess[[t, MARKET]].dropna()
        X = data[[MARKET]].values          # market excess return
        y = data[t].values                 # stock excess return
        model = LinearRegression().fit(X, y)
        raw_beta = model.coef_[0]
        rows.append({"Ticker": t, "Company": UNIVERSE[t],
                     "Raw beta": raw_beta,
                     "Adj beta": 0.67 * raw_beta + 0.33,   # Blume adjustment
                     "R2": model.score(X, y)})

    stats = pd.DataFrame(rows).set_index("Ticker")
    stats["Exp return"] = rf + stats["Adj beta"] * MRP     # CAPM
    cov_annual = returns.drop(columns=MARKET).cov() * 12
    return stats, cov_annual


# ---------------- Age -> risk band -> target beta ----------------
def age_risk_band(age):
    """Return (beta floor, beta ceiling) for an investor's age."""
    if age < 30:
        return 0.90, 1.50
    elif age < 45:
        return 0.80, 1.30
    elif age < 60:
        return 0.60, 1.10
    else:
        return 0.40, 0.90


def target_beta_from_slider(age, risk_pct):
    floor, ceiling = age_risk_band(age)
    return floor + (risk_pct / 100) * (ceiling - floor)


# ---------------- Pick stocks + optimize weights ----------------
def build_portfolio(stats, cov_annual, rf, age, risk_pct, amount, n_stocks):
    target = target_beta_from_slider(age, risk_pct)

    # pick the N stocks with beta closest to the target
    picks = (stats["Adj beta"] - target).abs().nsmallest(n_stocks).index.tolist()
    betas = stats.loc[picks, "Adj beta"].values
    exp_rets = stats.loc[picks, "Exp return"].values
    cov = cov_annual.loc[picks, picks].values

    # make sure the target is reachable within the weight limits
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n_stocks
    lo = linprog(betas, A_eq=[np.ones(n_stocks)], b_eq=[1], bounds=bounds).fun
    hi = -linprog(-betas, A_eq=[np.ones(n_stocks)], b_eq=[1], bounds=bounds).fun
    note = None
    if not (lo <= target <= hi):
        note = (f"Target beta {target:.2f} isn't reachable with these stocks, "
                f"so the closest possible ({np.clip(target, lo, hi):.2f}) is used.")
    target = float(np.clip(target, lo, hi))

    # minimize volatility subject to beta = target and weights summing to 1
    # (with CAPM returns, same beta = same return, so lowest volatility = highest Sharpe)
    def port_vol(w):
        return np.sqrt(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                   {"type": "eq", "fun": lambda w: w @ betas - target}]
    w0 = np.ones(n_stocks) / n_stocks
    result = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else w0

    port_beta = weights @ betas
    port_ret = weights @ exp_rets
    vol = port_vol(weights)
    sharpe = (port_ret - rf) / vol

    holdings = pd.DataFrame({
        "Ticker": picks,
        "Company": [UNIVERSE[t] for t in picks],
        "Beta": betas.round(2),
        "Weight %": (weights * 100).round(1),
        "$ Amount": (weights * amount).round(2),
    }).sort_values("Weight %", ascending=False).reset_index(drop=True)

    # rounding fix so totals are exactly 100% and the full amount
    holdings.loc[0, "Weight %"] = round(holdings.loc[0, "Weight %"] + 100 - holdings["Weight %"].sum(), 1)
    holdings.loc[0, "$ Amount"] = round(holdings.loc[0, "$ Amount"] + amount - holdings["$ Amount"].sum(), 2)

    summary = {"beta": port_beta, "vol": vol, "ret": port_ret, "sharpe": sharpe}
    return summary, holdings, note


# ---------------- Streamlit layout ----------------
st.set_page_config(page_title="Robo-Advisor", layout="wide")
st.title("Automated Financial Advisor")
st.caption("AC420 Project — Lydia Pelton & Sydney Pitts")

with st.sidebar:
    st.header("Your inputs")
    age = st.number_input("Age", min_value=18, max_value=100, value=22, step=1)
    floor, ceiling = age_risk_band(age)
    st.write(f"Risk range for your age: beta **{floor:.2f} – {ceiling:.2f}**")
    risk_pct = st.slider("Risk level (within your range)", 0, 100, 50,
                         help="0 = most conservative for your age, 100 = most aggressive")
    st.write(f"Target beta: **{target_beta_from_slider(age, risk_pct):.2f}**")
    amount = st.number_input("Amount to invest ($)", min_value=100, value=10_000, step=500)
    n_stocks = st.slider("Number of stocks", 10, 30, 15)

prices, rf = load_data()
stats, cov_annual = estimate_stats(prices, rf)
summary, holdings, note = build_portfolio(stats, cov_annual, rf, age, risk_pct, amount, n_stocks)

if note:
    st.info(note)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio beta (risk)", f"{summary['beta']:.2f}")
c2.metric("Volatility", f"{summary['vol']:.1%}")
c3.metric("Expected annual return", f"{summary['ret']:.1%}")
c4.metric("Sharpe ratio", f"{summary['sharpe']:.2f}")
st.caption(f"Risk-free rate (13-week T-bill): {rf:.2%} · Market risk premium assumed: {MRP:.1%}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Holdings")
    st.dataframe(
        holdings,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Weight %": st.column_config.NumberColumn(format="%.1f%%"),
            "$ Amount": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.write(f"**Total:** {holdings['Weight %'].sum():.1f}% · ${holdings['$ Amount'].sum():,.2f}")
with right:
    st.subheader("Allocation")
    fig = px.pie(holdings, names="Ticker", values="Weight %", hover_data=["Company", "$ Amount"])
    fig.update_traces(textposition="inside", textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("See every stock's beta and CAPM expected return"):
    st.dataframe(stats.sort_values("Adj beta").round(3), use_container_width=True)
