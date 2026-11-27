from Config import *
from Constraint_Builders import *
from Portfolio_Optimization import *
from Portfolio_Combination import *
import math
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
                          direction_to=direction_to)
        st.session_state["directions"] = (direction_from, direction_to)

    #show the last launched result (changed settings only apply after clicking Launch again)
    if "master" in st.session_state:
        master = st.session_state["master"]
        direction_from, direction_to = st.session_state["directions"]
        st.write(master)
        nice_graphs(master,direction_from=direction_from,direction_to=direction_to)

def nice_graphs (master:pl.DataFrame, direction_from, direction_to):
    #no copy needed: polars never changes master in place, every operation returns a new DataFrame
    #renaming the columns to make more sense

    fig_portfolio_value = px.line(master, x = "Date", y = "PORTFOLIO_VALUE", title="Portfolio CHF Value through time")
    st.plotly_chart(fig_portfolio_value)




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

    #ask for maximum weight: one cap for every asset, plus optional caps for single tickers
    max_w = st.toggle("Maximum weight per asset?")
    if max_w:
        cap_all = st.slider("Maximum weight for every asset (%)", 1, 100, value=20)
        #with a cap c, at least 1/c assets must be tradable on a date to reach 100 % (dates with fewer are skipped)
        st.caption(f"Needs at least {math.ceil(100 / cap_all)} tradable assets on a date, dates with fewer get no weights.")

        #table of the 41 tickers: leave the cell empty to keep the cap above, fill it to give that ticker its own cap.
        #the table never changes between reruns, so Streamlit keeps the edits (even when the slider moves)
        caps_table = pd.DataFrame({
            "Ticker": list(ASSETS.keys()) + list(STOCKS.keys()),
            "Group": ["Asset class"] * len(ASSETS) + ["Swiss stock"] * len(STOCKS),
            "Own cap (%)": [None] * (len(ASSETS) + len(STOCKS)),
        })
        with st.expander("Own cap for some tickers"):
            edited = st.data_editor(
                caps_table,
                key="caps_table",
                hide_index=True,
                disabled=["Ticker", "Group"],
                column_config={"Own cap (%)": st.column_config.NumberColumn(min_value=0, max_value=100, step=1)},
            )

        #one cap per ticker, by name (per_asset gives 100 % to tickers missing from a dict, so all 41 are filled in)
        caps = {
            row["Ticker"]: (cap_all if pd.isna(row["Own cap (%)"]) else row["Own cap (%)"]) / 100
            for row in edited.to_dict("records")
        }
        #same cap everywhere -> pass the number, otherwise the dict {ticker: cap}
        constraints["Max_Weight"] = cap_all / 100 if len(set(caps.values())) == 1 else caps

    #ask for financial amounts
    loan_amount = st.number_input(label="Loan amount", min_value=1, value=1_000_000)  # start amount of the portfolio
    interest = st.number_input(label="Interest payements", min_value=0.01)
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



    #refresh the data / launch the optimization: one wrapper call handles both buttons
    #(two calls would run the data refresh twice)
    refresh_data = st.button("Refresh Data")
    launch_click = st.button("Launch")
    wrapper(refresh_data=refresh_data, constraints=constraints, launch_click=launch_click,
            start_amount=loan_amount, interest=interest, frequency=withdrawal_frequency, taxes=taxes,
            direction_from=direction_from, direction_to=direction_to)

    #Show current constraint
    st.write(constraints)

    #Visuals for combination
    k = st.slider("K", 0,500)
    k = k/1000
    rebalance = st.slider("Rebalance schedule every _ month", 0,24, value=12 )

    years = st.slider("Years", 1,50, value=20)

    result = visual_combination(years, k, rebalance)
    fig = px.line(result, x="months", y=["ERC", "OPT"], )
    st.plotly_chart(fig)