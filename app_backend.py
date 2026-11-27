"""
app_backend.py  -  connects the Streamlit app (app.py) to the group's engine in Support_Files.

Part 1 runs the strategy with the engine's sequence() (Portfolio_Combination.py):
    master_df -> Combine_weights -> Portfolio_performance -> numerical_performance_portfolio
Part 2 only computes numbers for the charts (metrics, benchmarks, drawdowns, crises, efficient frontier).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent / "Support_Files"))

from Config import (ASSETS, STOCKS, TICKER_LIST, ASSETS_RET_FILE, STOCKS_RET_FILE,  # noqa: E402
                    RF_FILE, UNIVERSE_DATA_PATH)
from Constraint_Builders import (new_info_constraints, set_objective, add_long_only_click,  # noqa: E402
                                 add_target_ret, set_resampling, ERC_set_group_budget,
                                 new_constraint_dict, budget, long_only, add_equality)
from Portfolio_Optimization import Optimize_weights                                 # noqa: E402
from Portfolio_Combination import sequence                                          # noqa: E402

ASSET_NAMES = list(ASSETS.keys())        # the asset classes in Config.py
STOCK_NAMES = list(STOCKS.keys())        # the single Swiss stocks (SMI), optional in the app

def asset_class(ticker):
    """Group of each asset for the ERC risk budget and the charts."""
    if ticker in STOCKS:
        return "Swiss stocks"
    if "Gov" in ticker or "Corp" in ticker:
        return "Bonds"
    if ticker in ("Gold", "Silver"):
        return "Commodities"
    return "Equities"


ASSET_CLASS = {t: asset_class(t) for t in TICKER_LIST}
CLASS_ORDER = ["Equities", "Bonds", "Commodities", "Swiss stocks"]


def k_per_step(k_year, rebalance):
    """The engine's k bends the glide path once per rebalancing step. The app sets k per year, so the
    shape of the path stays the same whatever the rebalancing frequency: k_step = k_year * months / 12."""
    return k_year * rebalance / 12

CRISES = {
    "Dot-com crash (Sep 2000 - Sep 2002)": ("2000-09-01", "2002-09-30"),
    "Financial crisis (Nov 2007 - Feb 2009)": ("2007-11-01", "2009-02-28"),
    "Euro crisis (May - Sep 2011)": ("2011-05-01", "2011-09-30"),
    "SNB drops EUR floor (Jan 2015)": ("2015-01-01", "2015-01-31"),
    "Covid crash (Feb - Mar 2020)": ("2020-02-01", "2020-03-31"),
    "Inflation and rate hikes (Jan - Oct 2022)": ("2022-01-01", "2022-10-31"),
}


# ------------------------------------------------------------------------- part 1: the strategy
def run_backtest(s):
    """Run the whole strategy for the settings s chosen in the app. Returns pandas tables for the charts."""
    tickers = [t for t in TICKER_LIST if t in s["tickers"]]                         # Config order

    # constraints for the optimised portfolio (OPT), built with the functions of Constraint_Builders.py
    constraints = set_objective(new_info_constraints(), s["objective"])            # "MVP" or "Tangency"
    if s["long_only"]:
        constraints = add_long_only_click(constraints)
    # caps: None = no cap, a number = same cap for every asset, a dict {ticker: cap} = own cap per asset
    constraints["Max_Weight"] = s["opt_caps"]
    constraints["ERC_Cap"] = s["erc_caps"]
    if s["target_ret"] is not None:
        constraints = add_target_ret(constraints, "min", s["target_ret"] / 12)       # monthly excess return
    if s["michaud"] > 0:
        constraints = set_resampling(constraints, n_resamples=s["michaud"], seed=0)
    constraints = ERC_set_group_budget(constraints, s["group_budget"], ASSET_CLASS)   # ERC risk per class

    # the engine: OPT and ERC every month, glide path between them (shape k), returns,
    # client wealth after withdrawals and wealth tax
    direction_to = "ERC" if s["direction_from"] == "OPT" else "OPT"
    master = sequence(constraints=constraints, Tickers=tickers, start_date=s["start"], end_date=s["end"],
                      steps=1, min_obs=s["min_obs"], max_obs=s["rolling"],
                      start_amount=s["wealth"], interest=s["withdrawal"], interest_frequency=s["withdraw_every"],
                      taxes=s["tax"], direction_from=s["direction_from"], direction_to=direction_to,
                      k=k_per_step(s["k_year"], s["rebalance"]), rebalance_schedule=s["rebalance"])

    # from here: only reshaping into pandas tables for the charts
    m = master.sort("Date")
    dates = pd.to_datetime(m["Date"].to_list()) + pd.offsets.MonthEnd(0)
    perf = pd.DataFrame({"Portfolio": m["PORTFOLIO_PERFORMANCE"].to_numpy(),
                         "OPT": m["OPT_PERF"].to_numpy(), "ERC": m["ERC_PERF"].to_numpy()}, index=dates)
    value = pd.DataFrame({"Value": m["PORTFOLIO_VALUE"].to_numpy(), "Withdrawal": m["interest"].to_numpy(),
                          "Wealth tax": m["wealth_tax"].to_numpy()}, index=dates)

    w_opt = pd.DataFrame(m["OPT_weight"].to_list(), index=dates).fillna(0.0)[tickers]
    w_erc = pd.DataFrame(m["ERC_weight"].to_list(), index=dates).fillna(0.0)[tickers]
    share_erc = pd.Series(m["ERC_portfolio_weight"].to_numpy(), index=dates)       # glide path
    w_mix = w_opt.mul(1 - share_erc, axis=0) + w_erc.mul(share_erc, axis=0)

    # share of risk of each asset on the last date: w_i * (Sigma w)_i / (w' Sigma w)
    last = m.filter(pl.col("OPT_VCV").is_not_null()).tail(1).row(0, named=True)
    vcv, tk = np.array(last["OPT_VCV"], dtype=float), list(last["OPT_Tickers"])
    rc = {}
    for name, w in (("OPT", w_opt), ("ERC", w_erc), ("Portfolio", w_mix)):
        x = w.iloc[-1][tk].to_numpy()
        rc[name] = pd.Series(x * (vcv @ x) / (x @ vcv @ x), index=tk)

    return {"perf": perf, "value": value, "w_opt": w_opt, "w_erc": w_erc, "w_mix": w_mix,
            "share_erc": share_erc, "rc": pd.DataFrame(rc), "tickers": tickers}


# ----------------------------------------------------------------- part 2: numbers for the charts
def load_returns():
    """Monthly CHF total returns of the asset classes, CHF risk-free rate, availability table."""
    rets = pd.read_csv(ASSETS_RET_FILE, index_col=0, parse_dates=True)
    rf = pd.read_csv(RF_FILE, index_col=0, parse_dates=True)["RF_CHF"]
    av = pd.read_csv(UNIVERSE_DATA_PATH / "asset_availability.csv", index_col=0,
                     parse_dates=["first_return", "tradable_from"])
    return rets, rf, av.loc[av.index.isin(ASSET_NAMES)]


def first_returns():
    """First month with a return of every ticker (asset classes and Swiss stocks)."""
    frames = [pd.read_csv(f, index_col=0, parse_dates=True) for f in (ASSETS_RET_FILE, STOCKS_RET_FILE)]
    return pd.concat([f.apply(pd.Series.first_valid_index) for f in frames])


def estimated_start(first, tickers, start, min_obs, needed=1):
    """Rough first month of the backtest: when `needed` of the chosen assets have min_obs months of history
    inside the data period (a cap c needs at least 1/c tradable assets). Only used for the glide path preview."""
    start = pd.Timestamp(start)
    tradable = sorted(max(first[t], start) + pd.DateOffset(months=min_obs) for t in tickers if t in first)
    if not tradable:
        return None
    return tradable[min(needed, len(tradable)) - 1]


def glide_path(months, k_year, rebalance):
    """Share of the money in the target portfolio for each month, same formula and same rebalancing counter
    as Combine_weights in Portfolio_Combination.py (only for the preview, the backtest calls the engine)."""
    step = np.floor(np.arange(1, months + 1) / rebalance) + 1     # rebalancing period of each month
    t = step - step.min()
    if t.max() == 0:
        return np.zeros(months)
    k = k_per_step(k_year, rebalance)
    return t / t.max() if k == 0 else np.expm1(k * t) / np.expm1(k * t.max())


def metrics(r, rf):
    """Annual return, volatility, Sharpe (over the CHF risk-free rate), max drawdown, final value of 1."""
    r = r.dropna()
    if len(r) < 2:
        return {}
    wealth = (1 + r).cumprod()
    vol = r.std() * np.sqrt(12)
    excess = (r - rf.reindex(r.index).fillna(0)).mean() * 12
    return {"Return p.a.": wealth.iloc[-1] ** (12 / len(r)) - 1, "Volatility p.a.": vol,
            "Sharpe": excess / vol if vol > 0 else np.nan,
            "Max drawdown": (wealth / wealth.cummax() - 1).min(), "Growth of 1": wealth.iloc[-1]}


def benchmarks(index, rets, rf):
    """Simple alternatives on the same months as the backtest."""
    a = rets.reindex(index)
    return pd.DataFrame({"SPI (100% Swiss equity)": a["CH Equity"],
                         "60% SPI / 40% CH Gov 10y": 0.6 * a["CH Equity"] + 0.4 * a["CH Gov 10y"],
                         "S&P 500 in CHF": a["US Equity"], "CHF risk-free": rf.reindex(index)})


def drawdown(r):
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1


def crisis_table(perf):
    """Total return of each strategy during each crisis."""
    return pd.DataFrame({name: (1 + perf.loc[a:b]).prod() - 1 for name, (a, b) in CRISES.items()
                         if len(perf.loc[a:b])}).T


def frontier(rets, rf, tickers, date, min_obs=120, points=30):
    """Long-only efficient frontier on one date, with the engine's Optimize_weights (Ledoit-Wolf covariance)."""
    from sklearn.covariance import LedoitWolf
    ex = rets[tickers].sub(rf, axis=0).loc[:date]
    ok = [t for t in tickers if ex[t].notna().sum() >= min_obs]
    ex = ex[ok].dropna()
    if len(ok) < 2 or len(ex) < min_obs:
        return None
    S = LedoitWolf().fit(ex.to_numpy()).covariance_
    mu = ex.mean().to_numpy()
    curve = []
    for target in np.linspace(mu.min(), mu.max(), points):      # minimum variance for each target return
        cd = new_constraint_dict()
        budget(cd, len(ok))
        long_only(cd, len(ok))
        add_equality(mu, target, cd)
        w = Optimize_weights(S, cd)
        if w is not None:
            curve.append({"vol": np.sqrt(w @ S @ w * 12), "ret": w @ mu * 12})
    if not curve:
        return None
    c = pd.DataFrame(curve)
    return {"curve": c, "mvp": c.loc[c["vol"].idxmin()], "tan": c.loc[(c["ret"] / c["vol"]).idxmax()],
            "assets": pd.DataFrame({"vol": np.sqrt(np.diag(S) * 12), "ret": mu * 12}, index=ok),
            "obs": len(ex)}
