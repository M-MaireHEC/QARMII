import numpy as np
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from Config import *
from Portfolio_Optimization import *
from ERC import *
from Tools import *
from Constraint_Builders import *
from Data_Cleaner import ensure_master_data
import math
import plotly.express as px
from Wealth_Tax import wealth_base_tax, WEALTH_TAX_START

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
    master = master.with_columns(((pl.col("Rebalance")/(rebalance_schedule)).floor()+1).cast(pl.Int32).alias("Rebalance"))
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

        master = master.with_columns((pl.col("A")/pl.col("B")).alias(f"{direction_to}_portfolio_weight"))
    else:
        #A = t-t_min
        master = master.with_columns((pl.col("Rebalance")-pl.col("Rebalance").min()).alias("A"))
        #B = t_max - t_min
        master = master.with_columns((pl.col("Rebalance").max() - pl.col("Rebalance").min()).alias("B"))
        master = master.with_columns((pl.col("A")/pl.col("B")).alias(f"{direction_to}_portfolio_weight"))

    #Other portfolio weight
    master = master.with_columns((1-pl.col(f"{direction_to}_portfolio_weight")).alias(f"{direction_from}_portfolio_weight"))

    #Clean of columns A/B that are not useful
    master = master.drop("A", "B")

    fig = px.line(
            master.to_pandas(),
            x="Date",
            y=[f"{direction_to}_portfolio_weight",f"{direction_from}_portfolio_weight"],
            title="Portfolio Combination Weight"

        )

    #fig.show()

    return master

def Portfolio_performance(master):

    #call the master.parquet
    ensure_master_data()
    ret_df = pl.read_parquet(MASTER_FILE)

    #join the returns to the master_df
    #the join does not keep the date order, and cum_prod follows the row order
    master = master.join(ret_df, on="Date", how="inner").sort("Date")

    master = portfolio_drift_perf_handler(master,"OPT")
    master = portfolio_drift_perf_handler(master,"ERC")

    #compute the total portfolio return from both strategies (only once ERC and OPT are both built)
    #the ERC/OPT split is set at each rebalance and then drifts with each sleeve until the next one:
    #value since the rebalance = a * sum_ERC + b * sum_OPT, with a, b the shares of the period (a + b = 1)
    master = master.with_columns(
        (pl.col("ERC_portfolio_weight").first().over("Rebalance") * pl.col("sum_ERC")
         + pl.col("OPT_portfolio_weight").first().over("Rebalance") * pl.col("sum_OPT")).alias("sum_PORTFOLIO")
    )
    #return of month t = value_t / value_(t-1) - 1, the value before the first month of a period is 1
    master = master.with_columns(
        (pl.col("sum_PORTFOLIO") / pl.col("sum_PORTFOLIO").shift(1).over("Rebalance").fill_null(1) - 1).alias("PORTFOLIO_PERFORMANCE")
    )

    return master

def portfolio_drift_perf_handler(master, strategy):


    #tickers taken from the struct itself, so names always match the weights
    tick = [f.name for f in master.schema[f"{strategy}_weight"].fields]

    #Transform all returns into multipliers (missing return = 0%, so a null does not break the product)
    #Cummulative product of multipliers over reabalance
    master = master.with_columns((pl.col(*tick).fill_null(0)+1).cum_prod().over("Rebalance").name.suffix("_cumprod"))
    #multiply the weight with multiplier for OPT portfolio
    master = master.with_columns(
        (pl.col(f"{strategy}_weight").struct.field(t).first().over("Rebalance")  #take the ticker name from the struct
         *
         pl.col(f"{t}_cumprod")) #take the associated multiplier per ticker
        .alias(f"{t}_{strategy}_value") #rename to name_OPT_value once computation done.
    for t in tick #run over all the weights belonging to the struct
    )

    #Sum the ticker_cumprods to a sum column
    #row-wise sum (across tickers), no .over needed = portfolio value since the last rebalance
    TICKER_CUM_PROD = [f"{t}_{strategy}_value" for t in tick]
    master = master.with_columns(pl.sum_horizontal(TICKER_CUM_PROD).alias(f"sum_{strategy}"))

    #Normalize the drifted weights to get back to sum equal to 1 (parentheses: the suffix names the division)
    master = master.with_columns((pl.col(*TICKER_CUM_PROD)/pl.col(f"sum_{strategy}")).name.suffix("_normalized"))
    #check that sum of normalized weights equal to 1 (own name, sum_OPT is kept for the performance)
    TICKER_NORM = [f"{t}_normalized" for t in TICKER_CUM_PROD]
    master = master.with_columns(pl.sum_horizontal(TICKER_NORM).alias(f"check_{strategy}_weights"))

    # weights held at the start of month t = drifted weights at the end of t-1, per period;
    # the first month of each period starts from the target weight w0
    master = master.with_columns(
        pl.col(f"{t}_{strategy}_value_normalized").shift(1).over("Rebalance")
        .fill_null(pl.col(f"{strategy}_weight").struct.field(t).first().over("Rebalance"))
        .alias(f"{t}_{strategy}_w_start")
        for t in tick
    )

    # portfolio return of month t = sum over tickers of w_start * r_t
    master = master.with_columns(
        pl.sum_horizontal(
            pl.col(f"{t}_{strategy}_w_start") * pl.col(t).fill_null(0) for t in tick
        ).alias(f"{strategy}_PERF")
    )

    return master

#provide the numerical value of the portfolio taking into account interest payments
#interest_payment = amount paid out to the investor (taken from the portfolio) every interest_frequency months
#taxes=True: wealth tax (Wealth_Tax.py, coefficient 100 %) is taken from the portfolio on its value at 31 December
def numerical_performance_portfolio(master, start_amount:float, interest_payment:float,interest_frequency:int, taxes:bool= False):
    #only months where both portfolios have weights (otherwise PORTFOLIO_PERFORMANCE is a fake 0), in date order
    master = master.filter(pl.col("ERC_weight").is_not_null() & pl.col("OPT_weight").is_not_null()).sort("Date")

    #add the interest payment according to frequency
    master = master.with_row_index("Month",offset=1)
    master = master.with_columns(
        pl.when((pl.col("Month")%interest_frequency) == 0).then(pl.lit(float(interest_payment))).otherwise(0.00).alias("interest")
    )

    #value_t = value_(t-1) * (1 + return_t) - payment_t (- wealth tax in December)
    #polars cannot use the previous row of the column it is creating, so the recursion is a plain loop (one row per month)
    returns = master["PORTFOLIO_PERFORMANCE"].to_list()
    payments = master["interest"].to_list()
    is_december = (master["Date"].dt.month() == 12).to_list()

    value = float(start_amount)
    values, taxes_paid = [], []
    for ret, payment, december in zip(returns, payments, is_december):
        value = value * (1 + ret) - payment   #return of the month first, then the payment at the end of the month
        tax = 0.0
        if taxes and december and value >= WEALTH_TAX_START[0]:   #the tax table starts at 50'000 CHF
            tax = wealth_base_tax(value)
            value -= tax
        value = max(value, 0.0)   #portfolio used up: stays at 0
        values.append(value)
        taxes_paid.append(tax)

    master = master.with_columns(
        pl.Series("wealth_tax", taxes_paid),
        pl.Series("PORTFOLIO_VALUE", values),
    )
    print(master)
    return master


#Only for visuals for streamlit
def visual_combination(years, k, rebalancing):
    investment_period = years*12
    ranks = list(range(investment_period+1))
    df = pl.DataFrame({
        "months":ranks
    })
    df = df.with_columns(((pl.col("months")/(rebalancing)).floor()+1).alias("Rebalance"))

    if k != 0:
        # A == (e^(k*(year-min(years)))-1
        df = df.with_columns((((k * (pl.col("Rebalance") - pl.col("Rebalance").min())).exp()) - 1).alias("A"))
        # B == (e^(k*(max(year)-min(years)))-1
        df = df.with_columns(
            ((k * (pl.col("Rebalance").max() - pl.col("Rebalance").min())).exp() - 1).alias("B"))

        df = df.with_columns((pl.col("A") / pl.col("B")).alias(f"ERC"))
    else:
        # A = t-t_min
        df = df.with_columns((pl.col("Rebalance") - pl.col("Rebalance").min()).alias("A"))
        # B = t_max - t_min
        df = df.with_columns((pl.col("Rebalance").max() - pl.col("Rebalance").min()).alias("B"))
        df = df.with_columns((pl.col("A") / pl.col("B")).alias(f"ERC"))

    # Other portfolio weight
    df = df.with_columns((1 - pl.col(f"ERC")).alias(f"OPT"))
    return df


#Joins into one singular dataframe <!> Binary only support. Three ports will fail
def master_df(Ticker_list = TICKER_LIST, start_date = START_DATE, end_date = END_DATE, steps = STEPS, min_obs = MIN_OBS, max_obs = MAX_OBS, constraints = {}):
    #ERC portfolio

    ERC_dic = ERC_optimization(ticker_list=Ticker_list,start_date=start_date,end_date=end_date,step=steps,min_obs=min_obs,max_obs=max_obs)
    ERC_weights = ERC_dic["weights_df"]
    ERC_Sigma = ERC_dic["sigma_df"]
    del ERC_dic
    ERC_df = Summary_perf_df(ERC_weights,ERC_Sigma,Ticker_list)
    del ERC_weights,ERC_Sigma
    #Change "to ERC_{name}"
    ERC_df = ERC_df.rename(lambda column_name: "ERC_"+column_name)
    ERC_df = ERC_df.rename({"ERC_Date":"Date"})
    print(ERC_df.columns)

    #optimized portfolio
    opt_portfolio = rolling_weights(ticker_list=Ticker_list,constraints=constraints,start_date=start_date,end_date=end_date,step=steps,min_obs=min_obs,max_obs=max_obs)
    opt_weights = opt_portfolio["weights_df"]
    opt_Sigma = opt_portfolio["sigma_df"]
    del opt_portfolio
    opt_df = Summary_perf_df(opt_weights,opt_Sigma,Ticker_list)
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

def sequence(constraints = new_info_constraints(),
             Tickers = TICKER_LIST,
             start_date = START_DATE,
             end_date = END_DATE,
             steps = STEPS,
             min_obs = MIN_OBS,
             max_obs = MAX_OBS,
             start_amount = 1_000_000,
             interest = 2500,
             interest_frequency = 1,
             taxes = True,
             direction_from = "OPT",
             direction_to = "ERC",
             k = 1,
             rebalance_schedule = 1


):

    master = master_df(Ticker_list=Tickers,
                       start_date=start_date,
                       end_date=end_date,
                       steps=steps,
                       min_obs=min_obs,
                       max_obs=max_obs,
                       constraints=constraints,

                        )

    master = Combine_weights(master, k =  k, rebalance_schedule=rebalance_schedule, direction_from= direction_from, direction_to= direction_to  )
    master = Portfolio_performance(master=master)
    master = numerical_performance_portfolio(master = master,
                                             start_amount= start_amount,
                                             interest_payment = interest,
                                             interest_frequency=interest_frequency,
                                             taxes= taxes)

    return master



if "__main__" == __name__:

    #visual_combination(20,0,1)

    start = time.time()
    constraints = new_info_constraints()
    constraints = add_long_only_click(constraints)
    constraints = set_objective(constraints,"MVP")
    #constraints = set_resampling(constraints,2) #take som compute power, not activated for quicker tests.
    master = sequence(constraints=constraints, Tickers= TICKER_LIST, start_date=START_DATE, end_date=END_DATE, steps=STEPS,min_obs=MIN_OBS, max_obs=MAX_OBS)

    print(master.select("Date", "PORTFOLIO_PERFORMANCE", "interest", "wealth_tax", "PORTFOLIO_VALUE"))
    end = time.time()
    print(end-start, "Seconds")
