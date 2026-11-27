"""
QARM II - portfolio management web app for a Swiss long-term investor.
Run locally:  streamlit run app.py      (from the project folder)
The strategy code lives in Support_Files; app_backend.py connects it to this page.
"""
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

import app_backend as B
from Config import TICKER_LIST, END_DATE, MIN_OBS, MAX_OBS

st.set_page_config(page_title="Swiss Glide Path Portfolio", layout="wide")

_css = Path(__file__).with_name("style.css")      # custom look, see style.css
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------- styling
COLORS = {   # one fixed colour per entity, the same in every chart
    "Portfolio": "#257de0", "ERC": "#f47230", "OPT": "#00b894",
    "SPI (100% Swiss equity)": "#6a4ce2", "60% SPI / 40% CH Gov 10y": "#ee7fac",
    "S&P 500 in CHF": "#f2a600", "CHF risk-free": "#8c8c8c",
}
CLASS_COLORS = {"Equities": "#257de0", "Bonds": "#f47230", "Commodities": "#f2a600", "Swiss stocks": "#6a4ce2"}
PCT = "{:.2%}"


def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}%}"


def chf(x):
    return "n/a" if x is None or pd.isna(x) else f"CHF {x:,.0f}".replace(",", "'")


def chf_short(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"CHF {x / 1e6:.2f}m" if abs(x) >= 1e6 else f"CHF {x / 1e3:,.0f}k"


def style_fig(fig, height=420, yfmt=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15, title=None))
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


# --------------------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Preparing data...")
def data():
    return B.load_returns()


RETS, RF, AVAIL = data()
LAST_MONTH = RETS.index.max().date()


@st.cache_data(show_spinner=False, max_entries=30)
def cached_backtest(key):
    s = json.loads(key)
    s["start"] = pl.date(*map(int, s["start"].split("-")))
    s["end"] = pl.date(*map(int, s["end"].split("-")))
    return B.run_backtest(s)


@st.cache_data(show_spinner=False)
def cached_frontier(tickers, d, min_obs):
    return B.frontier(RETS, RF, list(tickers), d, min_obs=min_obs)


FIRST_RETURNS = st.cache_resource(B.first_returns)()


def glide_text(k, frm, to):
    """Plain words for the shape of the glide path."""
    if k == 0:
        return f"the same amount moves from {frm} to {to} at each rebalancing"
    if k > 0:
        return f"the money stays longer in {frm} and moves to {to} faster towards the end"
    return f"the money moves to {to} fast at the start, then more slowly"


# one row per asset, always all of them: the editor keeps the edits by row number, so the table never changes
CAPS_TABLE = pd.DataFrame({"Asset": TICKER_LIST,
                           "OPT cap (%)": np.full(len(TICKER_LIST), np.nan),
                           "ERC cap (%)": np.full(len(TICKER_LIST), np.nan)})


def caps_setting(edited, column, cap_all, tickers):
    """Caps for the engine: None (no cap), one number (same cap for every asset) or {ticker: cap}
    over the chosen assets only (the optimiser refuses caps for assets outside its ticker list)."""
    own = {r["Asset"]: r[column] for r in edited.to_dict("records") if pd.notna(r[column])}
    caps = {t: float(own.get(t, cap_all)) / 100 for t in tickers}
    values = set(caps.values())
    if all(v >= 1 for v in values):
        return None
    if len(values) == 1:
        return values.pop()
    return caps


def caps_total(caps, tickers):
    if caps is None:
        return float("inf")
    return sum(caps.values()) if isinstance(caps, dict) else caps * len(tickers)


# ------------------------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.title("Settings")

sb.subheader("Client")
wealth = sb.number_input("Starting wealth (CHF)", 100_000, 50_000_000, 1_000_000, step=100_000)
withdrawal = sb.number_input("Monthly withdrawal (CHF)", 0, 500_000, 2_500, step=500)
tax = sb.toggle("Pay Swiss wealth tax", value=True, help="Paid every December, 2025 base tax scale (Wealth_Tax.py).")

with sb.expander("Assets"):
    chosen_assets = st.multiselect("Asset classes", B.ASSET_NAMES, default=B.ASSET_NAMES)
    add_stocks = st.toggle("Add Swiss stocks (SMI)", value=False,
                           help=f"The {len(B.STOCK_NAMES)} single Swiss stocks. They form their own class "
                                "'Swiss stocks' for the ERC risk budget and the charts. Slower to run.")
    chosen_stocks = (st.multiselect("Swiss stocks", B.STOCK_NAMES, default=B.STOCK_NAMES) if add_stocks else [])
    tickers = [t for t in TICKER_LIST if t in set(chosen_assets) | set(chosen_stocks)]
    st.caption(f"{len(tickers)} of {len(TICKER_LIST)} assets selected.")

sb.subheader("Strategy: glide path")
objective = sb.radio(
    "Optimised portfolio (OPT)", ["MVP", "Tangency"], horizontal=True,
    help="MVP: the lowest-risk mix. Tangency: the best return per unit of risk (uses expected returns).")
direction = sb.radio("Direction", ["OPT → ERC", "ERC → OPT"], horizontal=True,
                     help="The portfolio the money starts in (100% at the start) and the one it ends in.")
direction_from = "OPT" if direction == "OPT → ERC" else "ERC"
direction_to = "ERC" if direction_from == "OPT" else "OPT"
k_year = sb.slider("Shape of the glide path, k (per year)", -0.5, 0.5, 0.0, 0.01, format="%.2f",
                   help="0: the same amount moves at each rebalancing (straight line). Above 0: the money "
                        "stays longer in the starting portfolio and moves late. Below 0: it moves early. "
                        "Set per year, so the shape does not change with the rebalancing frequency.")
rebalance = sb.slider("Rebalance every ... months", 1, 24, 1,
                      help="Back to the target weights, and one step along the glide path, every ... months.")
preview_slot = sb.container()        # glide path preview, drawn once the data period is known

with sb.expander("ERC: share of risk per asset class"):
    st.caption("ERC splits the risk between the asset classes as set here, then equally between the assets "
               "of each class. Equal = classic risk parity.")
    budget_classes = [c_ for c_ in B.CLASS_ORDER if any(B.ASSET_CLASS[t] == c_ for t in tickers)]
    group_budget = {c_: st.slider(c_, 5, 80, 33, 1, format="%d", key=f"gb_{c_}") for c_ in budget_classes}
    tot_b = sum(group_budget.values()) or 1
    st.caption("Share of ERC risk: " + ", ".join(f"{c_} {v / tot_b:.0%}" for c_, v in group_budget.items()))

with sb.expander("Maximum weight per asset"):
    st.caption("100% = no cap. With a cap c, at least 1/c assets must be tradable on a date: dates with fewer "
               "get no weights.")
    opt_cap = st.slider("OPT: cap for every asset", 5, 100, 100, 5, format="%d%%")
    erc_cap = st.slider("ERC: cap for every asset", 5, 100, 100, 5, format="%d%%")
    st.caption("Own cap for single assets: fill the cell, leave it empty to use the cap above. "
               "Assets that are not selected are ignored.")
    edited_caps = st.data_editor(
        CAPS_TABLE, key="caps_table", hide_index=True, disabled=["Asset"], height=300,
        column_config={c_: st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%d")
                       for c_ in ("OPT cap (%)", "ERC cap (%)")})
    st.caption("An ERC cap that binds makes the risk shares unequal: the result is then risk budgeting "
               "with bounds, not a pure ERC.")
opt_caps = caps_setting(edited_caps, "OPT cap (%)", opt_cap, tickers)
erc_caps = caps_setting(edited_caps, "ERC cap (%)", erc_cap, tickers)

with sb.expander("Advanced settings"):
    st.markdown("**Optimisation (OPT)**")
    long_only = st.toggle("Long only (no short selling)", value=True)
    use_target = st.toggle("Minimum expected excess return", value=False)
    target_ret = st.slider("Target (% p.a. above CHF risk-free)", 0.0, 8.0, 2.0, 0.25) / 100 if use_target else None
    michaud = st.slider("Michaud resampling (draws, 0 = off)", 0, 20, 0,
                        help="Averages the weights over simulated histories. Slower: about 1 s per draw.")
    st.markdown("**Estimation**")
    min_obs = st.select_slider("Months of history needed", [60, 120, 180], value=MIN_OBS)
    rolling = st.toggle("Rolling window (otherwise expanding)", value=bool(MAX_OBS))
    CONFIG_END = pl.select(END_DATE).item()          # the group's END_DATE in Config.py (a polars date)
    period = st.slider("Data period", date(1990, 1, 1), LAST_MONTH,
                       (date(1990, 1, 1), min(LAST_MONTH, CONFIG_END)), format="MMM YYYY",
                       help="Default end = END_DATE in Config.py. Data available until "
                            f"{LAST_MONTH:%b %Y}.")
    st.markdown("**Withdrawals**")
    withdraw_every = st.slider("Withdraw every ... months", 1, 36, 1,
                               help="The withdrawal above is paid at this frequency.")

with preview_slot:
    # rough start: when enough chosen assets have min_obs months of history (the run shows the real path)
    needed = max(math.ceil(100 / opt_cap), math.ceil(100 / erc_cap))
    p_start = B.estimated_start(FIRST_RETURNS, tickers, period[0], min_obs, needed)
    p_end = pd.Timestamp(period[1])
    if p_start is not None and p_start < p_end:
        p_dates = pd.date_range(p_start, p_end, freq="ME")
        share_to = B.glide_path(len(p_dates), k_year, rebalance)
        fig = go.Figure([go.Scatter(x=p_dates, y=1 - share_to, name=direction_from,
                                    line=dict(color=COLORS[direction_from], shape="hv")),
                         go.Scatter(x=p_dates, y=share_to, name=direction_to,
                                    line=dict(color=COLORS[direction_to], shape="hv"))])
        fig.update_layout(height=190, margin=dict(l=0, r=0, t=5, b=0), hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2, title=None))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"Preview, about {p_start:%Y} to {p_end:%Y}: {glide_text(k_year, direction_from, direction_to)}. "
                   "The Overview tab shows the path of the last run.")

# settings that cannot give a portfolio: say why and keep Run disabled
problems = []
if len(tickers) < 2:
    problems.append("Select at least 2 assets.")
for name_, caps_ in (("OPT", opt_caps), ("ERC", erc_caps)):
    if len(tickers) >= 2 and caps_total(caps_, tickers) < 1 - 1e-9:
        problems.append(f"The {name_} caps add up to {caps_total(caps_, tickers):.0%} over the selected assets: "
                        "the weights cannot reach 100%.")
for problem in problems:
    sb.warning(problem)

settings = dict(start=period[0].isoformat(), end=period[1].isoformat(), min_obs=min_obs, rolling=rolling,
                objective=objective, long_only=long_only, target_ret=target_ret,
                michaud=michaud, group_budget=group_budget, rebalance=rebalance,
                tickers=tickers, opt_caps=opt_caps, erc_caps=erc_caps,
                direction_from=direction_from, k_year=k_year,
                wealth=wealth, withdrawal=withdrawal, withdraw_every=withdraw_every, tax=tax)
key = json.dumps(settings, sort_keys=True)

run = sb.button("Run backtest", type="primary", width="stretch", disabled=bool(problems))
if not problems and (run or "result" not in st.session_state):
    try:
        with st.spinner("Optimising every month since 2000 (5 to 20 seconds, longer with Swiss stocks "
                        "or resampling)..."):
            st.session_state.result = cached_backtest(key)
            st.session_state.result_key = key
            st.session_state.result_settings = settings
            st.session_state.error = None
    except Exception as e:
        st.session_state.error = str(e)

st.title("Swiss Glide Path Portfolio")
st.caption("A long-term investment strategy for a Swiss investor: monthly total returns in CHF since 1990, "
           "an optimised portfolio and a risk-parity portfolio, blended along a glide path.")

if st.session_state.get("error"):
    st.error(f"The backtest could not run with these settings: {st.session_state.error}")
if "result" not in st.session_state:
    st.stop()
if st.session_state.result_key != key:
    st.info("Settings changed since the last run. Press **Run backtest** in the sidebar to update the results.")

R = st.session_state.result
S = st.session_state.result_settings
perf, value = R["perf"], R["value"]
bench = B.benchmarks(perf.index, RETS, RF)
allp = pd.concat([perf, bench], axis=1)
M = {c: B.metrics(allp[c], RF) for c in allp}
mp = M["Portfolio"]

tabs = st.tabs(["Overview", "Results", "Allocation", "Market explorer"])

# ----------------------------------------------------------------------------------- overview
def every(n):
    return "every month" if n == 1 else f"every {n} months"


gb_used = S.get("group_budget") or {}
gb_tot = sum(gb_used.values()) or 1
erc_text = ("every asset class carries the same share of risk, split equally between its assets"
            if len(set(gb_used.values())) <= 1 else
            "the risk is split between asset classes ("
            + ", ".join(f"{c_.lower()} {v / gb_tot:.0%}" for c_, v in gb_used.items())
            + "), then equally between the assets of each class")

FROM = S["direction_from"]
TO = "ERC" if FROM == "OPT" else "OPT"
opt_kind = "minimum-variance" if S["objective"] == "MVP" else "maximum-Sharpe"
why = (f"The client starts in the {'lower-risk' if S['objective'] == 'MVP' else 'optimised'} portfolio, when an "
       "early crash would hurt most because withdrawals lock in the losses, and ends in the broadly diversified "
       "risk-parity portfolio." if FROM == "OPT" else
       f"The client starts in the broadly diversified risk-parity portfolio and ends in the {opt_kind} portfolio.")
n_stocks = sum(t in B.STOCK_NAMES for t in R["tickers"])
universe = (f"{len(R['tickers']) - n_stocks} asset classes in CHF (equities, government and corporate bonds, "
            "gold, silver)" + (f" and {n_stocks} Swiss stocks" if n_stocks else ""))

with tabs[0]:
    st.subheader("The idea")
    st.markdown(
        f"""
- **Client:** a Swiss private investor with **{chf(S['wealth'])}**, withdrawing **{chf(S['withdrawal'])}**
  {every(S['withdraw_every'])}{', paying the wealth tax' if S['tax'] else ''}.
- **Two portfolios, re-estimated every month** from all history available at that date (no look-ahead):
  **OPT**, a {opt_kind} portfolio, and **ERC**, where {erc_text}.
- **Glide path:** over the investment period the money moves from **{FROM}** to **{TO}**:
  {glide_text(S['k_year'], FROM, TO)} (k = {S['k_year']:+.2f} per year, rebalancing {every(S['rebalance'])}).
  {why}
- **Universe:** {universe}. An asset enters once it has {S['min_obs']} months of history.
"""
    )
    c = st.columns(5)
    c[0].metric("Return p.a.", pct(mp["Return p.a."], 2))
    c[1].metric("Volatility p.a.", pct(mp["Volatility p.a."], 2))
    c[2].metric("Sharpe ratio", f"{mp['Sharpe']:.2f}")
    c[3].metric("Worst drawdown", pct(mp["Max drawdown"]))
    c[4].metric("Client wealth at the end", chf_short(value["Value"].iloc[-1]))
    spi = M.get("SPI (100% Swiss equity)")
    if spi:
        st.markdown(f"Over **{perf.index[0]:%b %Y} to {perf.index[-1]:%b %Y}** the strategy earned "
                    f"**{pct(mp['Return p.a.'], 2)}** a year with a worst loss of **{pct(mp['Max drawdown'])}**, "
                    f"against **{pct(spi['Return p.a.'], 2)}** and **{pct(spi['Max drawdown'])}** for Swiss "
                    f"equities (SPI).")
    g = (1 + allp[["Portfolio", "SPI (100% Swiss equity)", "60% SPI / 40% CH Gov 10y"]].fillna(0)).cumprod()
    fig = go.Figure([go.Scatter(x=g.index, y=g[cn], name=cn, line=dict(color=COLORS[cn])) for cn in g])
    fig.update_layout(title="Growth of CHF 1 (before withdrawals and tax)")
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    st.subheader(f"Glide path: how the money moves from {FROM} to {TO}")
    gp = pd.DataFrame({"ERC": R["share_erc"], "OPT": 1 - R["share_erc"]})[[FROM, TO]]
    fig = go.Figure([go.Scatter(x=gp.index, y=gp[c_], name=c_, line=dict(color=COLORS[c_], shape="hv"))
                     for c_ in gp])
    fig.update_layout(title="Share of the money in each portfolio during the backtest")
    st.plotly_chart(style_fig(fig, 340, ".0%"), width="stretch")

# ------------------------------------------------------------------------------------ results
with tabs[1]:
    st.subheader("Key figures")
    c = st.columns(4)
    c[0].metric("Return p.a.", pct(mp["Return p.a."], 2))
    c[1].metric("Volatility p.a.", pct(mp["Volatility p.a."], 2))
    c[2].metric("Sharpe ratio", f"{mp['Sharpe']:.2f}")
    c[3].metric("Worst drawdown", pct(mp["Max drawdown"]))
    c = st.columns(4)
    c[0].metric("Wealth at the end", chf_short(value["Value"].iloc[-1]))
    c[1].metric("Total withdrawn", chf_short(value["Withdrawal"].sum()))
    c[2].metric("Total wealth tax", chf_short(value["Wealth tax"].sum()))
    c[3].metric("Without withdrawals and tax", chf_short(S["wealth"] * mp["Growth of 1"]))

    st.subheader("Strategy against simple alternatives")
    show = st.multiselect("Lines", list(allp.columns), default=["Portfolio", "OPT", "ERC",
                                                                 "SPI (100% Swiss equity)",
                                                                 "60% SPI / 40% CH Gov 10y"])
    logy = st.toggle("Log scale", value=False)
    g = (1 + allp[show].fillna(0)).cumprod()
    fig = go.Figure([go.Scatter(x=g.index, y=g[cn], name=cn, line=dict(color=COLORS.get(cn))) for cn in show])
    fig.update_layout(title="Growth of CHF 1 (before withdrawals and tax)")
    if logy:
        fig.update_yaxes(type="log")
    st.plotly_chart(style_fig(fig), width="stretch")

    tbl = pd.DataFrame(M).T[["Return p.a.", "Volatility p.a.", "Sharpe", "Max drawdown", "Growth of 1"]]
    st.dataframe(tbl.style.format({"Return p.a.": PCT, "Volatility p.a.": PCT, "Sharpe": "{:.2f}",
                                   "Max drawdown": "{:.1%}", "Growth of 1": "{:.2f}x"}), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([go.Scatter(x=value.index, y=value["Value"], name="Client wealth",
                                    line=dict(color=COLORS["Portfolio"]))])
        fig.add_trace(go.Scatter(x=value.index, y=value["Withdrawal"].cumsum() + value["Wealth tax"].cumsum(),
                                 name="Paid out (withdrawals + tax), cumulative", line=dict(color="#8c8c8c")))
        fig.update_layout(title="The client's wealth, in CHF")
        st.plotly_chart(style_fig(fig, 380, ",.0f"), width="stretch")
    with c2:
        dd = pd.DataFrame({cn: B.drawdown(allp[cn]) for cn in ["Portfolio", "SPI (100% Swiss equity)",
                                                                "60% SPI / 40% CH Gov 10y"]})
        fig = go.Figure([go.Scatter(x=dd.index, y=dd[cn], name=cn, line=dict(color=COLORS[cn])) for cn in dd])
        fig.update_layout(title="Drawdown: loss from the previous peak")
        st.plotly_chart(style_fig(fig, 380, ".0%"), width="stretch")

    yr = (1 + allp[["Portfolio", "SPI (100% Swiss equity)", "60% SPI / 40% CH Gov 10y"]]).groupby(
        allp.index.year).prod() - 1
    yr = yr.reset_index(names="Year").melt("Year", var_name="Strategy", value_name="Return")
    fig = px.bar(yr, x="Year", y="Return", color="Strategy", barmode="group", color_discrete_map=COLORS)
    fig.update_layout(title="Calendar-year returns")
    st.plotly_chart(style_fig(fig, 380, ".0%"), width="stretch")

    st.subheader("Crises")
    ct = B.crisis_table(allp[["Portfolio", "OPT", "ERC", "SPI (100% Swiss equity)", "60% SPI / 40% CH Gov 10y"]])
    st.dataframe(ct.style.format("{:.1%}"), width="stretch")

# --------------------------------------------------------------------------------- allocation
with tabs[2]:
    sleeve = st.radio("Portfolio", ["Portfolio (blend)", "OPT", "ERC"], horizontal=True)
    W = {"Portfolio (blend)": R["w_mix"], "OPT": R["w_opt"], "ERC": R["w_erc"]}[sleeve]
    by_class = st.toggle("Group by asset class", value=True)
    if by_class:
        Wg = W.T.groupby(lambda t: B.ASSET_CLASS[t]).sum().T
        cols = [c_ for c_ in CLASS_COLORS if c_ in Wg]
        fig = go.Figure([go.Scatter(x=Wg.index, y=Wg[c_], name=c_, stackgroup="w",
                                    line=dict(width=0.5, color=CLASS_COLORS[c_])) for c_ in cols])
    else:
        top = W.mean().sort_values(ascending=False).index[:8].tolist()
        Wt = W[top].copy()
        Wt["Other"] = W.drop(columns=top).sum(axis=1)
        pal = px.colors.qualitative.Plotly
        fig = go.Figure([go.Scatter(x=Wt.index, y=Wt[c_], name=c_, stackgroup="w",
                                    line=dict(width=0.5, color=pal[i % 10] if c_ != "Other" else "#bdbdbd"))
                         for i, c_ in enumerate(Wt.columns)])
    fig.update_layout(title=f"{sleeve}: weights through time (target weights at each rebalancing)")
    st.plotly_chart(style_fig(fig, 420, ".0%"), width="stretch")

    c1, c2 = st.columns([1, 1])
    last = W.iloc[-1]
    last = last[last > 0.0005].sort_values(ascending=False)
    with c1:
        st.markdown(f"**Allocation on {W.index[-1]:%B %Y}** for {chf(S['wealth'])}, by asset class")
        byc = last.groupby(lambda t: B.ASSET_CLASS[t]).sum()
        byc = byc[[c_ for c_ in CLASS_COLORS if c_ in byc]]
        fig = go.Figure(go.Pie(labels=byc.index, values=byc.values, hole=0.55, sort=False,
                               marker=dict(colors=[CLASS_COLORS[c_] for c_ in byc.index],
                                           line=dict(color="white", width=2)),
                               textinfo="percent"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")
    with c2:
        out = pd.DataFrame({"Class": [B.ASSET_CLASS[t] for t in last.index], "Weight": last.values,
                            "Amount (CHF)": last.values * S["wealth"]}, index=last.index)
        st.dataframe(out.style.format({"Weight": "{:.1%}", "Amount (CHF)": "{:,.0f}"}), width="stretch",
                     height=380)

    st.subheader("Where does the risk come from?")
    rc_name = {"Portfolio (blend)": "Portfolio", "OPT": "OPT", "ERC": "ERC"}[sleeve]
    rc = R["rc"][rc_name].dropna()
    wc = W.iloc[-1].reindex(rc.index).fillna(0)
    byclass = pd.DataFrame({"Weight": wc.groupby(lambda t: B.ASSET_CLASS[t]).sum(),
                            "Share of risk": rc.groupby(lambda t: B.ASSET_CLASS[t]).sum()})
    byclass = byclass.reindex([c_ for c_ in CLASS_COLORS if c_ in byclass.index])
    fig = go.Figure([go.Bar(x=byclass.index, y=byclass["Weight"], name="Weight", marker_color="#257de0"),
                     go.Bar(x=byclass.index, y=byclass["Share of risk"], name="Share of risk",
                            marker_color="#f47230")])
    fig.update_layout(barmode="group", title=f"{sleeve}, last month: weight vs share of risk by asset class")
    st.plotly_chart(style_fig(fig, 360, ".0%"), width="stretch")
    wts = W.iloc[-1].reindex(rc.index).fillna(0)
    d = pd.DataFrame({"Weight": wts, "Share of risk": rc}).sort_values("Share of risk", ascending=False)
    d = d[(d["Weight"] > 0.001) | (d["Share of risk"].abs() > 0.001)].head(15)
    fig = go.Figure([go.Bar(x=d.index, y=d["Weight"], name="Weight", marker_color="#257de0"),
                     go.Bar(x=d.index, y=d["Share of risk"], name="Share of risk", marker_color="#f47230")])
    fig.update_layout(barmode="group", title="By asset, last month, 15 largest")
    st.plotly_chart(style_fig(fig, 400, ".0%"), width="stretch")

# ---------------------------------------------------------------------------- market explorer
with tabs[3]:
    st.subheader("The data")
    pick = st.multiselect("Assets", list(RETS.columns), default=["CH Equity", "US Equity", "CH Gov 10y", "Gold"])
    rng = st.slider("Period", date(1990, 1, 1), LAST_MONTH, (date(2000, 1, 1), LAST_MONTH), format="MMM YYYY",
                    key="mx_period")
    sub = RETS.loc[str(rng[0]):str(rng[1]), pick]
    if len(pick):
        g = (1 + sub.fillna(0)).cumprod()
        fig = px.line(g, color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(title="Growth of CHF 1, total return in CHF", yaxis_title=None, xaxis_title=None)
        st.plotly_chart(style_fig(fig, 400), width="stretch")
        st.dataframe(pd.DataFrame({c_: B.metrics(sub[c_], RF) for c_ in pick}).T.style.format(
            {"Return p.a.": PCT, "Volatility p.a.": PCT, "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
             "Growth of 1": "{:.2f}x"}), width="stretch")

    st.subheader("Correlations")
    cc = RETS.loc[str(rng[0]):str(rng[1]), B.ASSET_NAMES].corr()
    fig = px.imshow(cc, zmin=-1, zmax=1, text_auto=".1f", aspect="auto",
                    color_continuous_scale=[[0, "#257de0"], [0.5, "#f2f2f2"], [1, "#f47230"]])
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Monthly return correlations, {rng[0]:%b %Y} to {rng[1]:%b %Y}")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Efficient frontier on a chosen date")
    fdate = st.select_slider("Date", [d_.date() for d_ in pd.date_range("2000-12-31", LAST_MONTH, freq="YE")],
                             value=pd.Timestamp(min(LAST_MONTH, date(2024, 12, 31))).date())
    f = cached_frontier(tuple(t for t in R["tickers"] if t in B.ASSET_NAMES), str(fdate), S["min_obs"])
    if f is None:
        st.info("Not enough history on that date.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f["curve"]["vol"], y=f["curve"]["ret"], name="Efficient frontier (long only)",
                                 line=dict(color="#257de0")))
        fig.add_trace(go.Scatter(x=f["assets"]["vol"], y=f["assets"]["ret"], mode="markers+text",
                                 text=f["assets"].index, textposition="top center", name="Assets",
                                 marker=dict(size=9, color="#8c8c8c")))
        fig.add_trace(go.Scatter(x=[f["mvp"]["vol"]], y=[f["mvp"]["ret"]], mode="markers", name="MVP",
                                 marker=dict(size=14, color="#00b894", symbol="diamond")))
        fig.add_trace(go.Scatter(x=[f["tan"]["vol"]], y=[f["tan"]["ret"]], mode="markers", name="Tangency",
                                 marker=dict(size=14, color="#f47230", symbol="star")))
        fig.update_layout(title=f"Estimated with {f['obs']} months up to {fdate:%b %Y} (excess returns over CHF "
                                f"risk-free, Ledoit-Wolf covariance)", xaxis_title="Volatility p.a.",
                          yaxis_title="Expected excess return p.a.", hovermode="closest")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, 520, ".0%"), width="stretch")

    st.subheader("When each asset class enters the universe")
    av = AVAIL.copy()
    av["end"] = pd.Timestamp(LAST_MONTH)
    fig = go.Figure()
    for t, r_ in av.sort_values("first_return", ascending=False).iterrows():
        fig.add_trace(go.Scatter(x=[r_["first_return"], r_["end"]], y=[t, t], mode="lines",
                                 line=dict(color="#bdbdbd", width=8), showlegend=False, hoverinfo="skip"))
        if pd.notna(r_["tradable_from"]):
            fig.add_trace(go.Scatter(x=[r_["tradable_from"], r_["end"]], y=[t, t], mode="lines",
                                     line=dict(color="#257de0", width=8), showlegend=False,
                                     hovertemplate=f"{t}: data from {r_['first_return']:%b %Y}, "
                                                   f"tradable from {r_['tradable_from']:%b %Y}<extra></extra>"))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10),
                      title="Grey: data available. Blue: tradable (120 months of history)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Data: monthly total returns converted to CHF. Indices and funds from Investing.com, Yahoo Finance "
               "and MSCI, exchange rates and the CHF risk-free rate from FRED. Sources and processing are listed "
               "in DATA.md in the GitHub repository.")

# ------------------------------------------------------------------------------------ compare
with tabs[1]:
    st.divider()
    st.subheader("Compare scenarios")
    st.markdown("Pin the current result, change the settings, run again and pin again.")
    if "pins" not in st.session_state:
        st.session_state.pins = {}
    c1, c2 = st.columns([3, 1])
    label = c1.text_input("Name for this scenario",
                          value=f"{FROM} to {TO} ({S['objective']}), k {S['k_year']:+.2f}, rebalance "
                                f"{S['rebalance']}m, ERC risk " + "/".join(str(v) for v in S["group_budget"].values()))
    if c2.button("Pin current result", width="stretch"):
        st.session_state.pins[label] = {"metrics": mp, "growth": (1 + perf["Portfolio"]).cumprod(),
                                        "end": value["Value"].iloc[-1]}
    if st.session_state.pins:
        tb = pd.DataFrame({k_: {**v["metrics"], "Client wealth at the end": v["end"]}
                           for k_, v in st.session_state.pins.items()}).T
        st.dataframe(tb.drop(columns=["Growth of 1"]).style.format(
            {"Return p.a.": PCT, "Volatility p.a.": PCT, "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
             "Client wealth at the end": "{:,.0f}"}), width="stretch")
        fig = go.Figure([go.Scatter(x=v["growth"].index, y=v["growth"], name=k_)
                         for k_, v in st.session_state.pins.items()])
        fig.update_layout(title="Growth of CHF 1 (before withdrawals and tax)")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
        if st.button("Clear pinned scenarios"):
            st.session_state.pins = {}
            st.rerun()
