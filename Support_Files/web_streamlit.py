from Config import *
from Constraint_Builders import *
from Portfolio_Optimization import *
from Portfolio_Combination import *
import plotly.express as px
import streamlit as st


if __name__ == "__main__":
    st.title("Web Streamlit")
    long = st.toggle("Long only portfolio")
    st.write(long)

    k = st.slider("K", 0,500)
    k = k/1000
    rebalance = st.slider("Rebalance schedule every _ month", 0,24, value=12 )

    years = st.slider("Years", 1,50, value=20)

    result = visual_combination(years, k, rebalance)
    fig = px.line(result, x="months", y=["ERC", "OPT"], )
    st.plotly_chart(fig)