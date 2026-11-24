import pandas as pd
from Config import *
from Portfolio_Optimization import *
from ERC import *
from Tools import *
from Constraint_Builders import *



"""
Combination formula bounded:

A/B,
Where A = (e^(k*(year-min(years)))-1
Where B = (e^(k*(year-max(years)))-1
Where k is parameter.
K = 0 gives linear
K > 0 gives exponential concave
k > 0 gives exponential convex



"""
#Joins into one singular dataframe <!> Binary only support. Three ports will fail
def master_df(Ticker_list = TICKER_LIST, start_date = START_DATE, end_date = END_DATE, steps = STEPS, min_obs = MIN_OBS, max_obs = MAX_OBS, constraints = {}):
    #ERC portfolio

    ERC_dic = ERC_optimization(ticker_list=TICKER_LIST,start_date=START_DATE,end_date=END_DATE,step=STEPS,min_obs=MIN_OBS,max_obs=MAX_OBS)
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
    constraints = set_objective(constraints,"Tangency")
    constraints = set_resampling(constraints,20)

    master_df(Ticker_list = TICKER_LIST,start_date=START_DATE,end_date=END_DATE,steps=STEPS,min_obs=MIN_OBS,max_obs=MAX_OBS,constraints=constraints)

