"""
app_backend.py  -  glue between the Streamlit app (app.py) and the strategy code in Support_Files.

Only calls the existing functions (rolling_sigmas, weights_for_date, michaud_weights, build_date_args,
Summary_perf_df, Combine_weights, Portfolio_performance, numerical_performance_portfolio, ...).
New here: client views (exclude / less / more / must-have), risk budgets for ERC, benchmarks,
risk contributions, efficient frontier, crisis table and the "will the money last" simulation.
Nothing in Support_Files is modified.
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Support_Files"))

import plotly.graph_objects as go                       # noqa: E402
go.Figure.show = lambda *args, **kwargs: None           # charts are drawn by the app, never in a browser tab

from Config import UNIVERSE_DATA_PATH, MASTER_DATA_PATH, ASSETS, STOCKS          # noqa: E402
from Tools import rolling_sigmas, Create_excess_return_df, Summary_perf_df         # noqa: E402
from Constraint_Builders import (add_inequality, add_equality, new_constraint_dict,  # noqa: E402
                                 budget, long_only, to_cvx, VALID_OBJECTIVES)
from Portfolio_Optimization import (weights_for_date, michaud_weights,              # noqa: E402
                                    build_date_args, Optimize_weights)
from ERC import CustomPrior                                                         # noqa: E402
from skfolio.optimization import RiskBudgeting                                      # noqa: E402
from Portfolio_Combination import (Combine_weights, Portfolio_performance,          # noqa: E402
                                   numerical_performance_portfolio, visual_combination)
from Wealth_Tax import wealth_base_tax, WEALTH_TAX_START                            # noqa: E402
import Data_Cleaner                                                                 # noqa: E402

ASSET_NAMES = list(ASSETS.keys())
STOCK_NAMES = list(STOCKS.keys())

ASSET_CLASS = {   # the classes used for ERC and for the charts
    "Gold": "Commodities", "Silver": "Commodities",
    "US Gov 10y": "Bonds", "US Gov 30y": "Bonds", "CH Gov 2y": "Bonds", "CH Gov 10y": "Bonds",
    "DE Gov 10y": "Bonds", "UK Gov 10y": "Bonds", "US Corp IG": "Bonds", "US Corp HY": "Bonds",
    "US Equity": "Equities", "CH Equity": "Equities", "DE Equity": "Equities", "FR Equity": "Equities",
    "UK Equity": "Equities", "US Small Cap": "Equities",
}
for _s in STOCK_NAMES:
    ASSET_CLASS[_s] = "Swiss stocks"

VIEWS = ["Exclude", "Less", "Neutral", "More", "Must-have"]
VIEW_RULES = {   # OPT: weight floor / cap,  ERC: risk budget multiplier
    "Less": {"cap": 0.05, "budget": 0.5},
    "Neutral": {"budget": 1.0},
    "More": {"floor": 0.05, "budget": 2.0},
    "Must-have": {"floor": 0.10, "budget": 3.0},
}

CRISES = {
    "Dot-com crash (Sep 2000 - Sep 2002)": ("2000-09-01", "2002-09-30"),
    "Financial crisis (Nov 2007 - Feb 2009)": ("2007-11-01", "2009-02-28"),
    "Euro crisis (May - Sep 2011)": ("2011-05-01", "2011-09-30"),
    "SNB drops EUR floor (Jan 2015)": ("2015-01-01", "2015-01-31"),
    "Covid crash (Feb - Mar 2020)": ("2020-02-01", "2020-03-31"),
    "Inflation and rate hikes (Jan - Oct 2022)": ("2022-01-01", "2022-10-31"),
}


# ----------------------------------------------------------------------------------------- data
def ensure_master():
    """Build Data_Output/Master/Master.parquet from the 4 CSVs if needed (it is not on GitHub).
    Also writes 'master.parquet' (lower case), the name Portfolio_performance reads: on Linux
    (Streamlit Cloud) the two names are different files."""
    names = set(os.listdir(MASTER_DATA_PATH)) if MASTER_DATA_PATH.exists() else set()
    if "Master.parquet" not in names:
        MASTER_DATA_PATH.mkdir(parents=True, exist_ok=True)
        Data_Cleaner.pandas_csv_to_polars_cleaner(
            path=UNIVERSE_DATA_PATH / "smi_stocks_returns_chf_monthly.csv",
            rf_path=UNIVERSE_DATA_PATH / "rf_chf_monthly.csv",
            other_asset_path=UNIVERSE_DATA_PATH / "assets_returns_chf_monthly.csv")
        names = set(os.listdir(MASTER_DATA_PATH))
    if "master.parquet" not in names and not (MASTER_DATA_PATH / "master.parquet").exists():
        pl.read_parquet(MASTER_DATA_PATH / "Master.parquet").write_parquet(MASTER_DATA_PATH / "master.parquet")


def load_returns():
    """Asset classes + Swiss stocks (monthly CHF total returns), CHF risk-free, availability table."""
    a = pd.read_csv(UNIVERSE_DATA_PATH / "assets_returns_chf_monthly.csv", index_col=0, parse_dates=True)
    s = pd.read_csv(UNIVERSE_DATA_PATH / "smi_stocks_returns_chf_monthly.csv", index_col=0, parse_dates=True)
    rf = pd.read_csv(UNIVERSE_DATA_PATH / "rf_chf_monthly.csv", index_col=0, parse_dates=True)["RF_CHF"]
    av = pd.read_csv(UNIVERSE_DATA_PATH / "asset_availability.csv", index_col=0,
                     parse_dates=["first_return", "tradable_from"])
    return a.join(s, how="outer"), rf, av


# ------------------------------------------------------------------------- views -> constraints
def view_limits(tickers, views, overrides, global_cap):
    """Per ticker: floor and cap for OPT, risk budget for ERC."""
    floors, caps, budgets = {}, {}, {}
    for t in tickers:
        rule = VIEW_RULES.get(views.get(t, "Neutral"), {})
        lo = rule.get("floor", 0.0)
        hi = min(rule.get("cap", 1.0), global_cap)
        o = overrides.get(t, {})
        if o.get("min") is not None:
            lo = o["min"]
        if o.get("max") is not None:
            hi = o["max"]
        floors[t], caps[t] = lo, max(hi, lo)
        budgets[t] = rule.get("budget", 1.0)
    return floors, caps, budgets


def add_floors(date_args, floors, tangency):
    """Minimum weights w_i >= f_i (new). Tangency works on y = scaled weights: f_i * sum(y) - y_i <= 0."""
    args = copy.deepcopy(date_args)
    n = len(floors)
    ones = np.ones(n)
    for i, f in enumerate(floors):
        if f <= 0:
            continue
        e = np.zeros(n)
        e[i] = 1.0
        add_inequality(-e, -f, args["cons_w"])                     # w-form, used for the feasibility checks
        if tangency:
            add_inequality(f * ones - e, 0.0, args["cons"])
        else:
            add_inequality(-e, -f, args["cons"])
    return args


# ----------------------------------------------------------------------------- optimisations
def opt_weights(tickers, s, floors_all, caps_all, sigma_df):
    """Same loop as Portfolio_Optimization.rolling_weights, plus minimum weights per asset."""
    N = len(tickers)
    tangency = s["objective"] == "Tangency"
    constraints = {"Long_only": s["long_only"], "Max_Weight": None, "Relative_Weight": None,
                   "Target_Ret": ({"kind": "min", "target": s["target_ret"] / 12}
                                  if s["target_ret"] is not None else None),
                   "Resampling": None, "Objective": s["objective"]}
    Z = None
    if s["michaud"] > 0:
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((s["michaud"], int(sigma_df["Obs_num"].max()), N))

    cache, rows = {}, []
    for row in sigma_df.iter_rows(named=True):
        weight = None
        if row["VCV"] is not None:
            tk = row["Tickers"]
            pos = [tickers.index(t) for t in tk]
            fl = np.array([floors_all[t] for t in tk])
            cp = np.array([caps_all[t] for t in tk])
            if fl.sum() > 1:                       # impossible: scale the floors down to 100 %
                fl = fl / fl.sum()
            if cp.sum() < 1:                       # impossible: caps cannot reach 100 %, relax them
                cp = np.minimum(1.0, cp / cp.sum())
            key = (tuple(tk), tuple(fl), tuple(cp))
            if key not in cache:
                base = build_date_args(constraints, len(tk), None if np.all(cp >= 1) else cp,
                                       tangency, constraints["Target_Ret"])
                cache[key] = add_floors(base, fl, tangency)
            args = cache[key]
            sigma = np.array(row["VCV"], dtype=float)
            mu = np.array(row["MU"], dtype=float)
            if Z is None:
                w = weights_for_date(sigma, mu, **args)
            else:
                w = michaud_weights(sigma, mu, row["Obs_num"], Z, **args)
            if w is not None:
                weight = np.zeros(N)
                weight[pos] = w
        rows.append({"Date": row["Date"], "Weights": None if weight is None else weight.tolist()})
    return pl.DataFrame(rows, schema={"Date": pl.Date, "Weights": pl.List(pl.Float64)})


def erc_weights(tickers, s, budgets_all, floors_all, caps_all, excess_df, sigma_df):
    """Same loop as ERC.ERC_optimization, plus risk budgets and min / max weights per asset."""
    N = len(tickers)
    rows = []
    for row in sigma_df.iter_rows(named=True):
        if row["VCV"] is None:
            rows.append({"Date": row["Date"], "Weights": None})
            continue
        tk = row["Tickers"]
        window = (excess_df.filter((pl.col("Date") >= s["start"]) & (pl.col("Date") <= row["Date"]))
                  .sort("Date").select("Date", *tk).drop_nulls())
        if s["rolling"]:
            window = window.tail(s["min_obs"])
        rb = np.array([budgets_all[t] for t in tk], dtype=float)          # client views (1 = neutral)
        if s.get("erc_mode", "groups") == "groups":     # risk split between asset classes, then equally inside
            grp = s.get("erc_groups", ASSET_CLASS)
            gb = s.get("group_budget") or {}             # e.g. {"Equities": 25, ...}; missing class = 1 share
            gs = [grp[t] for t in tk]
            count = {g: gs.count(g) for g in set(gs)}
            total = sum(gb.get(g, 1.0) for g in count)   # classes not tradable yet are left out
            rb = rb * np.array([gb.get(g, 1.0) / total / count[g] for g in gs])
        lo = np.array([floors_all[t] if s["erc_follow_limits"] else 0.0 for t in tk])
        hi = np.array([caps_all[t] if s["erc_follow_limits"] else 1.0 for t in tk])
        if lo.sum() > 1:
            lo = lo / lo.sum()
        if hi.sum() < 1:
            hi = np.minimum(1.0, hi / hi.sum())
        try:
            model = RiskBudgeting(risk_budget=rb / rb.sum(), min_weights=lo, max_weights=hi,
                                  prior_estimator=CustomPrior(np.array(row["VCV"], dtype=float)))
            model.fit(window.select(tk).to_numpy())
            w = np.zeros(N)
            w[[tickers.index(t) for t in tk]] = model.weights_
            rows.append({"Date": row["Date"], "Weights": w.tolist()})
        except Exception:
            rows.append({"Date": row["Date"], "Weights": None})
    return pl.DataFrame(rows, schema={"Date": pl.Date, "Weights": pl.List(pl.Float64)})


def build_master(tickers, s, views, overrides):
    """Same as Portfolio_Combination.master_df, with the client's views."""
    floors, caps, budgets = view_limits(tickers, views, overrides, s["max_weight"])
    excess = Create_excess_return_df(tickers)
    sigma_df = rolling_sigmas(excess, start_date=s["start"], end_date=s["end"], step=1,
                              min_obs=s["min_obs"], max_obs=s["rolling"])
    if sigma_df.height == 0 or sigma_df["VCV"].is_null().all():
        raise ValueError(f"No date has {s['min_obs']} months of data: choose a longer period or a shorter window.")

    parts = {}
    for name, wdf in (("ERC", erc_weights(tickers, s, budgets, floors, caps, excess, sigma_df)),
                      ("OPT", opt_weights(tickers, s, floors, caps, sigma_df))):
        df = Summary_perf_df(wdf, sigma_df, tickers)
        df = df.rename(lambda c: f"{name}_{c}").rename({f"{name}_Date": "Date"})
        parts[name] = df
    master = parts["OPT"].join(parts["ERC"], on="Date", how="inner")
    return master.filter(~pl.all_horizontal(pl.all().exclude("Date").is_null())), sigma_df


# ------------------------------------------------------------------------------- full backtest
def run_backtest(s, views, overrides):
    """Everything the app shows for one set of settings. Returns plain pandas / python objects."""
    ensure_master()
    tickers = [t for t in s["universe"] if views.get(t, "Neutral") != "Exclude"]
    if len(tickers) < 2:
        raise ValueError("Keep at least two assets in the universe.")

    master, sigma_df = build_master(tickers, s, views, overrides)
    frm, to = ("OPT", "ERC") if s["direction"] == "OPT -> ERC" else ("ERC", "OPT")
    master = Combine_weights(master, s["k"], s["rebalance"], direction_from=frm, direction_to=to)
    master = Portfolio_performance(master=master)
    master = numerical_performance_portfolio(master, s["wealth"], s["withdrawal"], s["withdraw_every"], s["tax"])

    m = master.sort("Date")
    dates = pd.to_datetime(m["Date"].to_list()) + pd.offsets.MonthEnd(0)
    perf = pd.DataFrame({"Portfolio": m["PORTFOLIO_PERFORMANCE"].to_numpy(),
                         "OPT": m["OPT_PERF"].to_numpy(), "ERC": m["ERC_PERF"].to_numpy()}, index=dates)
    value = pd.DataFrame({"Value": m["PORTFOLIO_VALUE"].to_numpy(), "Withdrawal": m["interest"].to_numpy(),
                          "Wealth tax": m["wealth_tax"].to_numpy()}, index=dates)

    # target weights: OPT and ERC sleeves, mixed with this period's split
    def struct_df(col):
        return pd.DataFrame(m[col].to_list(), index=dates).fillna(0.0)[tickers]
    w_opt, w_erc = struct_df("OPT_weight"), struct_df("ERC_weight")
    share_erc = pd.Series(m["ERC_portfolio_weight"].to_numpy(), index=dates)
    w_mix = w_opt.mul(1 - share_erc, axis=0) + w_erc.mul(share_erc, axis=0)

    # risk contributions on the last date
    last = m.filter(pl.col("OPT_VCV").is_not_null()).tail(1).row(0, named=True)
    vcv = np.array(last["OPT_VCV"], dtype=float)
    tk_last = list(last["OPT_Tickers"])
    rc = {}
    for name, w in (("OPT", w_opt), ("ERC", w_erc), ("Portfolio", w_mix)):
        x = w.iloc[-1][tk_last].to_numpy()
        var = x @ vcv @ x
        rc[name] = pd.Series(x * (vcv @ x) / var if var > 0 else np.nan, index=tk_last)

    return {"perf": perf, "value": value, "w_opt": w_opt, "w_erc": w_erc, "w_mix": w_mix,
            "share_erc": share_erc, "rc": pd.DataFrame(rc), "tickers": tickers,
            "first_weights": sigma_df.filter(pl.col("VCV").is_not_null())["Date"].min()}


# ------------------------------------------------------------------------------------ analytics
def metrics(r, rf):
    """Annual return, volatility, Sharpe (over the CHF risk-free rate), max drawdown, final value of 1."""
    r = r.dropna()
    if len(r) < 2:
        return {}
    wealth = (1 + r).cumprod()
    ann = wealth.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    ex = (r - rf.reindex(r.index).fillna(0)).mean() * 12
    dd = (wealth / wealth.cummax() - 1).min()
    return {"Return p.a.": ann, "Volatility p.a.": vol, "Sharpe": ex / vol if vol > 0 else np.nan,
            "Max drawdown": dd, "Growth of 1": wealth.iloc[-1]}


def benchmarks(index, rets, rf):
    """Simple comparisons on the same months as the backtest."""
    a = rets.reindex(index)
    out = {}
    if "CH Equity" in a:
        out["SPI (100% Swiss equity)"] = a["CH Equity"]
        if "CH Gov 10y" in a:
            out["60% SPI / 40% CH Gov 10y"] = 0.6 * a["CH Equity"] + 0.4 * a["CH Gov 10y"]
    if "US Equity" in a:
        out["S&P 500 in CHF"] = a["US Equity"]
    out["CHF risk-free"] = rf.reindex(index)
    return pd.DataFrame(out)


def drawdown(r):
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1


def crisis_table(perf):
    rows = {}
    for name, (a, b) in CRISES.items():
        p = perf.loc[a:b]
        if len(p):
            rows[name] = (1 + p).prod() - 1
    return pd.DataFrame(rows).T


def frontier(rets, rf, tickers, date, min_obs=120, points=30):
    """Long-only efficient frontier on one date (excess returns, Ledoit-Wolf, expanding window)."""
    from sklearn.covariance import LedoitWolf
    ex = rets[tickers].sub(rf, axis=0).loc[:date]
    ok = [t for t in tickers if ex[t].notna().sum() >= min_obs]
    ex = ex[ok].dropna()
    if len(ok) < 2 or len(ex) < min_obs:
        return None
    S = LedoitWolf().fit(ex.to_numpy()).covariance_
    mu = ex.mean().to_numpy()
    n = len(ok)
    curve = []
    for target in np.linspace(mu.min(), mu.max(), points):
        cd = new_constraint_dict()
        budget(cd, n)
        long_only(cd, n)
        add_equality(mu, target, cd)
        w = Optimize_weights(S, cd)
        if w is not None:
            curve.append({"vol": np.sqrt(w @ S @ w * 12), "ret": w @ mu * 12, "w": w})
    if not curve:
        return None
    c = pd.DataFrame(curve)
    mvp = c.loc[c["vol"].idxmin()]
    tan = c.loc[(c["ret"] / c["vol"]).idxmax()]
    assets = pd.DataFrame({"vol": np.sqrt(np.diag(S) * 12), "ret": mu * 12}, index=ok)
    return {"curve": c, "mvp": mvp, "tan": tan, "assets": assets, "tickers": ok, "obs": len(ex)}


def money_lasts(monthly_returns, wealth, withdrawal, every, years, tax, n_paths=2000, block=12, seed=0):
    """Future wealth paths by resampling the strategy's own monthly returns in 12-month blocks."""
    r = monthly_returns.dropna().to_numpy()
    T = years * 12
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, len(r) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :T]   # consecutive 12-month blocks
    paths = r[idx]
    v = np.full(n_paths, float(wealth))
    out = np.empty((n_paths, T))
    for t in range(T):
        v = v * (1 + paths[:, t])
        if (t + 1) % every == 0:
            v = v - withdrawal
        if tax and (t + 1) % 12 == 0:
            taxable = v >= WEALTH_TAX_START[0]
            v[taxable] -= np.array([wealth_base_tax(x) for x in v[taxable]])
        v = np.maximum(v, 0.0)
        out[:, t] = v
    q = np.percentile(out, [5, 25, 50, 75, 95], axis=0)
    fan = pd.DataFrame(q.T, columns=["5%", "25%", "50%", "75%", "95%"], index=np.arange(1, T + 1) / 12)
    ruined = (out[:, -1] <= 0).mean()
    first_zero = np.where((out <= 0).any(axis=1), (out <= 0).argmax(axis=1) / 12, np.nan)
    return {"fan": fan, "p_ruin": ruined, "median_end": np.median(out[:, -1]),
            "median_years_if_ruined": np.nanmedian(first_zero) if ruined > 0 else None}
