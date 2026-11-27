from Config import *
from Constraint_Builders import *
from Portfolio_Optimization import *
from Portfolio_Combination import *
import math
from typing import Final
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fetch_universe import *
from Data_Cleaner import *

def wrapper(refresh_data:bool = False,
            constraints:dict = new_info_constraints(),
            launch_click : bool = False,
            start_amount: float = 1_000_000,
            interest = 1_000,
            frequency = 1,
            taxes : bool = False,
            direction_from = "OPT",
            direction_to = "ERC",
            k = 1,
            rebalance_schedule = 1,
            ):
    if refresh_data:
        #fetch the data
        build()
        #move to master
        pandas_csv_to_polars_cleaner()

    #Launch the Optimizazion when the Launch button is clicked.
    #st.button is True for one rerun only, so the result is kept in st.session_state:
    #otherwise it disappears as soon as another widget (K slider, ...) reruns the script.
    if launch_click:
        print("OPT launched")
        st.session_state["master"] = sequence(constraints=constraints,
                          Tickers=TICKER_LIST,
                          start_amount=start_amount,
                          interest_frequency= frequency,
                          interest = interest,
                          taxes = taxes,
                          direction_from = direction_from,
                          direction_to=direction_to,
                            k = k,
                            rebalance_schedule = rebalance_schedule,)
        st.session_state["directions"] = (direction_from, direction_to)

    #show the last launched result (changed settings only apply after clicking Launch again)
    if "master" in st.session_state:
        master = st.session_state["master"]
        direction_from, direction_to = st.session_state["directions"]
        nice_graphs(master,direction_from=direction_from,direction_to=direction_to)


#----- Visuals -----
#41 assets are too many to tell apart by colour, so the allocation charts show 6 asset groups
#(the weights of every single asset are in the table under the charts).
GROUP_ORDER: Final = ["Government bonds", "Corporate bonds", "Precious metals",
                      "Foreign equity", "Swiss equity index", "Swiss stocks"]
#categorical colours in a fixed order (validated reference palette): a group keeps its colour in every chart
GROUP_COLORS: Final = dict(zip(GROUP_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
PORTFOLIO_NAMES: Final = {"OPT": "Optimized", "ERC": "ERC"}
PORTFOLIO_COLORS: Final = {"Optimized": "#6250d6", "ERC": "#e34948"}  # other slots than the groups, so no colour means two things

def asset_group(ticker: str) -> str:
    if ticker in STOCKS:
        return "Swiss stocks"
    if ticker == "CH Equity":
        return "Swiss equity index"
    if "Gov" in ticker:
        return "Government bonds"
    if "Corp" in ticker:
        return "Corporate bonds"
    if ticker in ("Gold", "Silver"):
        return "Precious metals"
    return "Foreign equity"

def swiss_number(x: float) -> str:
    return f"{x:,.0f}".replace(",", "'")

#weights actually held at the start of each month inside one portfolio (target weight at the rebalance, then drifted)
def held_weights(master: pl.DataFrame, strategy: str) -> pl.DataFrame:
    tick = [f.name for f in master.schema[f"{strategy}_weight"].fields]
    return master.select("Date", *[pl.col(f"{t}_{strategy}_w_start").alias(t) for t in tick])

#value of each portfolio at the start of the month, per 1 CHF invested at the last rebalance:
#its share at the rebalance times its growth since then (same rule as sum_PORTFOLIO in Portfolio_performance)
def start_of_month_values():
    erc = pl.col("ERC_portfolio_weight").first().over("Rebalance") * pl.col("sum_ERC").shift(1).over("Rebalance").fill_null(1.0)
    opt = pl.col("OPT_portfolio_weight").first().over("Rebalance") * pl.col("sum_OPT").shift(1).over("Rebalance").fill_null(1.0)
    return erc, opt

#share of the Optimized and of the ERC portfolio in the total, at the start of each month
def portfolio_split(master: pl.DataFrame) -> pl.DataFrame:
    erc, opt = start_of_month_values()
    return master.select("Date", (opt / (erc + opt)).alias("Optimized"), (erc / (erc + opt)).alias("ERC"))

#total asset allocation: each asset's weight in both portfolios, weighted by the portfolios' shares
#(sum over assets of weight * return gives back PORTFOLIO_PERFORMANCE)
def total_allocation(master: pl.DataFrame) -> pl.DataFrame:
    erc, opt = start_of_month_values()
    tick = [f.name for f in master.schema["OPT_weight"].fields]
    return master.select("Date", *[
        ((erc * pl.col(f"{t}_ERC_w_start") + opt * pl.col(f"{t}_OPT_w_start")) / (erc + opt)).alias(t)
        for t in tick
    ])

#sums the asset weights of each group, per date (groups that stay at 0 are dropped)
def weights_by_group(weights: pl.DataFrame) -> pl.DataFrame:
    long = weights.unpivot(index="Date", variable_name="Asset", value_name="Weight")
    long = long.with_columns(pl.col("Asset").replace_strict({t: asset_group(t) for t in TICKER_LIST}).alias("Group"))
    grouped = long.group_by("Date", "Group").agg(pl.col("Weight").sum()).sort("Date")
    used = grouped.group_by("Group").agg(pl.col("Weight").abs().max()).filter(pl.col("Weight") > 1e-6)["Group"]
    return grouped.filter(pl.col("Group").is_in(used.to_list()))

#stacked allocation through time. Stacked areas cannot show short positions, so bars are used when a group goes negative
def allocation_chart(weights: pl.DataFrame, title: str):
    df = weights_by_group(weights)
    has_shorts = (df["Weight"] < -1e-9).any()
    chart = px.bar if has_shorts else px.area
    fig = chart(df, x="Date", y="Weight", color="Group", title=title,
                category_orders={"Group": GROUP_ORDER}, color_discrete_map=GROUP_COLORS)
    if has_shorts:
        fig.update_layout(barmode="relative", bargap=0)
        fig.update_traces(marker_line_width=0)
    else:
        fig.update_traces(line_width=1)
        fig.for_each_trace(lambda tr: tr.update(fillcolor=tr.line.color))  # solid fill, same colour as the bars
    fig.update_traces(hovertemplate="%{y:.1%}")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None,
                      legend_title_text="", hovermode="x unified")
    return fig

#table of every asset: average and latest weight in each portfolio and in the total (numbers in %)
def asset_weight_table(opt_w: pl.DataFrame, erc_w: pl.DataFrame, total_w: pl.DataFrame) -> pd.DataFrame:
    def stats(weights, name):
        long = weights.unpivot(index="Date", variable_name="Asset", value_name="w")
        last_date = weights["Date"].max()
        return long.group_by("Asset").agg(
            (pl.col("w").mean() * 100).alias(f"{name} avg"),
            (pl.col("w").filter(pl.col("Date") == last_date).first() * 100).alias(f"{name} latest"),
        )
    table = (stats(total_w, "Total")
             .join(stats(opt_w, "Optimized"), on="Asset")
             .join(stats(erc_w, "ERC"), on="Asset")
             .with_columns(pl.col("Asset").replace_strict({t: asset_group(t) for t in TICKER_LIST}).alias("Group")))
    numbers = [c for c in table.columns if c.endswith((" avg", " latest"))]
    table = table.filter(pl.any_horizontal(pl.col(numbers).abs() > 1e-4))   # assets never held are left out
    return table.select("Asset", "Group", *numbers).sort("Total avg", descending=True).to_pandas()

#realised performance from the monthly returns in CHF (before withdrawals and taxes)
def performance_summary(master: pl.DataFrame) -> pd.DataFrame:
    rf = master["RF_CHF"].fill_null(0).to_numpy()
    rows = []
    for name, column in (("Optimized", "OPT_PERF"), ("ERC", "ERC_PERF"), ("Combined", "PORTFOLIO_PERFORMANCE")):
        r = master[column].to_numpy()
        excess = r - rf
        wealth = np.cumprod(1 + r)
        peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]   # the start value 1 counts as a peak
        rows.append({
            "Portfolio": name,
            "Return p.a.": r.mean() * 12 * 100,
            "CAGR": (wealth[-1] ** (12 / len(r)) - 1) * 100,
            "Volatility p.a.": r.std(ddof=1) * np.sqrt(12) * 100,
            "Sharpe": excess.mean() / excess.std(ddof=1) * np.sqrt(12),
            "Max drawdown": (wealth / peak - 1).min() * 100,
            "Worst month": r.min() * 100,
            "Best month": r.max() * 100,
        })
    return pd.DataFrame(rows)

def nice_graphs (master:pl.DataFrame, direction_from, direction_to):
    #no copy needed: polars never changes master in place, every operation returns a new DataFrame
    st.header("Results")
    start, end = master["Date"].min(), master["Date"].max()

    #headline numbers of the investor's account
    tile_value, tile_paid, tile_tax, tile_period = st.columns(4)
    tile_value.metric("Final value (CHF)", swiss_number(master["PORTFOLIO_VALUE"][-1]))
    tile_paid.metric("Withdrawn (CHF)", swiss_number(master["interest"].sum()))
    tile_tax.metric("Wealth tax paid (CHF)", swiss_number(master["wealth_tax"].sum()))
    tile_period.metric("Months invested", master.height)

    st.subheader("Performance summary")
    st.caption(f"Monthly returns in CHF from {start:%Y-%m} to {end:%Y-%m}, before withdrawals and taxes. "
               "Sharpe ratio uses the CHF risk-free rate. Optimized and ERC are each shown as if held alone.")
    percent = st.column_config.NumberColumn(format="%.2f %%")
    summary = performance_summary(master)
    st.dataframe(summary, hide_index=True,
                 column_config={c: percent for c in summary.columns if c not in ("Portfolio", "Sharpe")} | {
                     "Sharpe": st.column_config.NumberColumn(format="%.2f")})

    fig_portfolio_value = px.line(master, x = "Date", y = "PORTFOLIO_VALUE", title="Portfolio value (CHF), after withdrawals and taxes")
    fig_portfolio_value.update_traces(line_width=2, hovertemplate="%{x|%Y-%m}: %{y:,.0f} CHF")
    fig_portfolio_value.update_layout(yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig_portfolio_value)

    #allocations
    st.subheader("Asset allocation")
    opt_w = held_weights(master, "OPT")
    erc_w = held_weights(master, "ERC")
    total_w = total_allocation(master)
    tab_total, tab_split, tab_opt, tab_erc = st.tabs(
        ["Total allocation", "Optimized vs ERC", "Optimized portfolio", "ERC portfolio"])
    with tab_total:
        st.plotly_chart(allocation_chart(total_w, "Total asset allocation (start of each month)"))
    with tab_split:
        split = portfolio_split(master).unpivot(index="Date", variable_name="Portfolio", value_name="Share")
        fig_split = px.area(split, x="Date", y="Share", color="Portfolio",
                            title=f"Share of each portfolio: from {PORTFOLIO_NAMES[direction_from]} to {PORTFOLIO_NAMES[direction_to]}",
                            category_orders={"Portfolio": ["Optimized", "ERC"]}, color_discrete_map=PORTFOLIO_COLORS)
        fig_split.update_traces(line_width=1, hovertemplate="%{y:.1%}")
        fig_split.for_each_trace(lambda tr: tr.update(fillcolor=tr.line.color))
        fig_split.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None,
                                legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig_split)
    with tab_opt:
        st.plotly_chart(allocation_chart(opt_w, "Optimized portfolio: weights by asset group"))
    with tab_erc:
        st.plotly_chart(allocation_chart(erc_w, "ERC portfolio: weights by asset group"))

    with st.expander("Weights of every asset"):
        st.caption(f"Average over {start:%Y-%m} to {end:%Y-%m}, and latest month ({end:%Y-%m}).")
        st.dataframe(asset_weight_table(opt_w, erc_w, total_w), hide_index=True,
                     column_config={f"{p} {s}": st.column_config.NumberColumn(format="%.1f %%")
                                    for p in ("Total", "Optimized", "ERC") for s in ("avg", "latest")})

    with st.expander("Raw results table"):
        st.write(master)




#Maximum weight input, used for both portfolios: one cap for every asset, plus optional caps for single tickers.
#Returns None (no cap), a number (same cap for every ticker) or a dict {ticker: cap}.
def max_weight_input(portfolio: str, table_key: str, note: str = ""):
    if not st.toggle(f"Maximum weight per asset ({portfolio})?"):
        return None
    cap_all = st.slider(f"Maximum weight for every asset, {portfolio} (%)", 1, 100, value=20)
    #with a cap c, at least 1/c assets must be tradable on a date to reach 100 % (dates with fewer are skipped)
    st.caption(f"Needs at least {math.ceil(100 / cap_all)} tradable assets on a date, dates with fewer get no weights. {note}")

    #table of the 41 tickers: leave the cell empty to keep the cap above, fill it to give that ticker its own cap.
    #the table never changes between reruns, so Streamlit keeps the edits (even when the slider moves)
    caps_table = pd.DataFrame({
        "Ticker": list(ASSETS.keys()) + list(STOCKS.keys()),
        "Group": ["Asset class"] * len(ASSETS) + ["Swiss stock"] * len(STOCKS),
        "Own cap (%)": [None] * (len(ASSETS) + len(STOCKS)),
    })
    with st.expander(f"Own cap for some tickers ({portfolio})"):
        edited = st.data_editor(
            caps_table,
            key=table_key,
            hide_index=True,
            disabled=["Ticker", "Group"],
            column_config={"Own cap (%)": st.column_config.NumberColumn(min_value=0, max_value=100, step=1)},
        )

    #one cap per ticker, by name (tickers missing from a dict get 100 %, so all 41 are filled in)
    caps = {
        row["Ticker"]: (cap_all if pd.isna(row["Own cap (%)"]) else row["Own cap (%)"]) / 100
        for row in edited.to_dict("records")
    }
    #same cap everywhere -> pass the number, otherwise the dict {ticker: cap}
    return cap_all / 100 if len(set(caps.values())) == 1 else caps


if __name__ == "__main__":
    #portfolio settings (Long_only, Max_Weight, ...), not new_constraint_dict() which holds the solver matrices A, b, G, h
    constraints = new_info_constraints()

    st.title("Web Streamlit")

    #Add long only constraint
    long = st.toggle("Long only portfolio")
    #Streamlit reruns the whole script on every click and rebuilds constraints, so set the value instead of flipping it
    constraints["Long_only"] = long

    #Asks the objective
    objective = st.segmented_control(
        label="Objective",
        options=["MVP", "Tangency"],
        default="MVP",
        label_visibility="collapsed"
    )
    if objective == "Tangency":
        constraints = set_objective(constraints, "Tangency")
    else:
        constraints = set_objective(constraints, "MVP")


    #Ask for required return
    ret_min = st.toggle("Required min return?")
    if ret_min:
        ret_value = st.number_input("What is the minimum return?")
        kind = st.segmented_control(label="Kind", options=["exact", "min"], default="min")
        constraints = add_target_ret(constraints, kind = kind, target=ret_value)

    #ask for ressampling
    ressample = st.toggle("Sample resample?")
    if ressample:
        sample_val = st.number_input("What is the number of samples? (recommended low number)",min_value=1)
        constraints = set_resampling(constraints, sample_val)

    #ask for maximum weights, separately for the optimized portfolio and for the ERC portfolio
    constraints["Max_Weight"] = max_weight_input("optimized portfolio", table_key="caps_table")
    constraints["ERC_Cap"] = max_weight_input(
        "ERC portfolio", table_key="erc_caps_table",
        note="A cap that binds makes the risk contributions unequal: the result is then risk budgeting with bounds, not a pure ERC.")

    #ask for financial amounts
    loan_amount = st.number_input(label="Loan amount", min_value=1, value=1_000_000)  # start amount of the portfolio
    interest = st.number_input(label="Interest payements", min_value=0, value = 600)
    additional_spending = st.number_input(label="Additional withdrawals", min_value=0)
    interest = interest+additional_spending
    withdrawal_frequency = st.slider("How frequently do you withdraw ?", min_value=1, max_value=36, value = 12)
    taxes = st.toggle("Account for taxes?")

    #ask from what portfolio to start
    Combination = st.segmented_control(
        label="Combination",
        options=[f"{objective} -> ERC", f"ERC -> {objective}"],
        default=f"{objective} -> ERC",
        label_visibility="collapsed"
    )

    if Combination == f"{objective} -> ERC":
        direction_from = "OPT"
        direction_to = "ERC"
    else:
        direction_from = "ERC"
        direction_to = "OPT"


    #Visuals for combination
    k = st.slider("K", 0,500)
    k = k/1000
    rebalance = st.slider("Rebalance schedule every _ month", 1,24, value=12 )  # 0 would divide by zero

    years = st.slider("Years", 1,50, value=20)

    result = visual_combination(years, k, rebalance)
    fig = px.line(result, x="months", y=["ERC", "OPT"], )
    st.plotly_chart(fig)

    #refresh the data / launch the optimization: one wrapper call handles both buttons
    #(two calls would run the data refresh twice)
    refresh_data = st.button("Refresh Data")
    launch_click = st.button("Launch")
    wrapper(refresh_data=refresh_data, constraints=constraints, launch_click=launch_click,
            start_amount=loan_amount, interest=interest, frequency=withdrawal_frequency, taxes=taxes,
            direction_from=direction_from, direction_to=direction_to, k = k, rebalance_schedule=rebalance)

    #Show current constraint
    st.write(constraints)

