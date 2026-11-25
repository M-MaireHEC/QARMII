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

def wrapper(refresh_data:bool = True):
    if refresh_data:
        #fetch the data
        build()
        #move to master
        pandas_csv_to_polars_cleaner()





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


    st.write(constraints)

    #refresh the data
    refresh_data = st.button("Refresh Data")
    wrapper(refresh_data)

    #Visuals for combination
    k = st.slider("K", 0,500)
    k = k/1000
    rebalance = st.slider("Rebalance schedule every _ month", 0,24, value=12 )

    years = st.slider("Years", 1,50, value=20)

    result = visual_combination(years, k, rebalance)
    fig = px.line(result, x="months", y=["ERC", "OPT"], )
    st.plotly_chart(fig)