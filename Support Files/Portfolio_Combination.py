import numpy as np
import pandas as pd
from Config import *
from Portfolio_Optimization import *
from ERC import *
from Tools import *
from Constraint_Builders import *
import math
import plotly.express as px

VALID_DIRECTIONS:Final = ["ERC","OPT"]

"""
Combination formula bounded:

A/B,
Where A = (e^(k*(year-min(years)))-1
Where B = (e^(k*(max(years)-min(years)))-1
Where k is parameter.
K = 0 gives linear
K > 0 gives exponential concave
k > 0 gives exponential convex



"""
#Rebalance is month until rebalance occur. 12 == yearly.
def Combine_weights(master, k, rebalance_schedule, direction_from = "OPT", direction_to = "ERC"):
    direction_from = direction_from.upper()
    direction_to = direction_to.upper()

    if direction_from not in VALID_DIRECTIONS:
        direction_from = "OPT"
    if direction_to not in VALID_DIRECTIONS:
        direction_to = "ERC"


    total_obs = min(len(master["ERC_weight"]), len(master["OPT_weight"])) #maximum common found weights
    #slice from below
    master = master[-(total_obs-1):]

    #add counter row
    master = master.with_row_index("Rebalance", offset = 1)
    #adapt to rebalance schedule
    master = master.with_columns(((pl.col("Rebalance")/(rebalance_schedule+1)).floor()+1).cast(pl.Int32).alias("Rebalance"))
    print(master)

    #Ffill in case of nonetype or others in columns
    master = master.with_columns([
        pl.col("ERC_weight").forward_fill(),
        pl.col("OPT_weight").forward_fill()
    ])



    if k != 0:
        # A == (e^(k*(year-min(years)))-1
        master = master.with_columns((((k*(pl.col("Rebalance")-pl.col("Rebalance").min())).exp())-1).alias("A"))
        # B == (e^(k*(max(year)-min(years)))-1
        master = master.with_columns(((k*(pl.col("Rebalance").max()-pl.col("Rebalance").min())).exp()-1).alias("B"))

        master = master.with_columns((pl.col("A")/pl.col("B")).alias(f"{direction_to}_weight"))
    else:
        #A = t-t_min
        master = master.with_columns((pl.col("Rebalance")-pl.col("Rebalance").min()).alias("A"))
        #B = t_max - t_min
        master = master.with_columns((pl.col("Rebalance").max() - pl.col("Rebalance").min()).alias("B"))
        master = master.with_columns((pl.col("A")/pl.col("B")).alias(f"{direction_to}_weight"))

    #Other portfolio weight
    master = master.with_columns((1-pl.col(f"{direction_to}_weight")).alias(f"{direction_from}_weight"))

    #Clean of columns A/B that are not useful
    master = master.remove(pl.col("A"), pl.col("B"))

    fig = px.line(
            master.to_pandas(),
            x="Date",
            y=[f"{direction_to}_weight",f"{direction_from}_weight"],
            title="Portfolio Combination Weight"

        )

    fig.show()

    return master

def Portfolio_performance(master):
    pass

#Joins into one singular dataframe <!> Binary only support. Three ports will fail
def master_df(Ticker_list = TICKER_LIST, start_date = START_DATE, end_date = END_DATE, steps = STEPS, min_obs = MIN_OBS, max_obs = MAX_OBS, constraints = {}):
    #ERC portfolio

    ERC_dic = ERC_optimization(ticker_list=Ticker_list,start_date=start_date,end_date=end_date,step=steps,min_obs=min_obs,max_obs=max_obs)
    ERC_weights = ERC_dic["weights_df"]
    ERC_Sigma = ERC_dic["sigma_df"]
    del ERC_dic
    ERC_df = Summary_perf_df(ERC_weights,ERC_Sigma)
    del ERC_weights,ERC_Sigma
    #Change "to ERC_{name}"
    ERC_df = ERC_df.rename(lambda column_name: "ERC_"+column_name)
    ERC_df = ERC_df.rename({"ERC_Date":"Date"})
    print(ERC_df.columns)

    #optimized portfolio
    opt_portfolio = rolling_weights(ticker_list=TICKER_LIST,constraints=constraints,start_date=START_DATE,end_date=END_DATE,step=STEPS,min_obs=MIN_OBS,max_obs=MAX_OBS)
    opt_weights = opt_portfolio["weights_df"]
    opt_Sigma = opt_portfolio["sigma_df"]
    del opt_portfolio
    opt_df = Summary_perf_df(opt_weights,opt_Sigma)
    del opt_weights,opt_Sigma
    #Change to OPT_{name}
    opt_df = opt_df.rename(lambda column_name: "OPT_"+column_name)
    opt_df = opt_df.rename({"OPT_Date":"Date"})
    print(opt_df.columns)

    #Join DF on the common date
    master = opt_df.join(ERC_df,on=["Date"],how="inner")
    del ERC_df, opt_df

    #Removing the rows with nulls (fully null)
    master = master.filter(~pl.all_horizontal(pl.all().exclude("Date").is_null()))

    return master



if "__main__" == __name__:
    constraints = new_info_constraints()
    constraints = add_long_only_click(constraints)
    constraints = set_objective(constraints,"MVP")
    constraints = set_resampling(constraints,2)

    master = master_df(Ticker_list = TICKER_LIST,start_date=START_DATE,end_date=END_DATE,steps=STEPS,min_obs=MIN_OBS,max_obs=MAX_OBS,constraints=constraints)
    Combine_weights(master, 0.1,12 )

