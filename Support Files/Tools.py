import numpy as np
from cvxopt import matrix, solvers
from typing import Final
import polars as pl
import sklearn
from sklearn.covariance import LedoitWolf
from Config import *
from skfolio.optimization import RiskBudgeting
from skfolio.prior import BasePrior, ReturnDistribution
from Constraint_Builders import *

#Preparing the Sigmas per period (method uses LedoitWolf).
def rolling_sigmas(excess_return_df:pl.DataFrame, start_date, end_date, step = 1, min_obs = 24, max_obs:bool = False):
    # max_obs=True -> rolling window of min_obs months; max_obs=False -> expanding window (all months since start)
    excess_return_df = excess_return_df.filter(
        (pl.col("Date")>=start_date) & (pl.col("Date")<=end_date)
    )

    excess_return_df = excess_return_df.sort("Date").with_row_index("Order")

    n = excess_return_df.shape[0]

    results = []
    schema = pl.Schema({
        "Date": pl.Date,
        "VCV": pl.List(pl.List(pl.Float64)),
        "MU": pl.List(pl.Float64),  # one mean excess return per ticker (a flat list, not a matrix)
        "Obs_num":pl.Int64,
        "Tickers": pl.List(str)
    })


    # Each window ends at row end_month (exclusive): the first one holds min_obs rows,
    # then the end moves forward by `step` rows until the last row (n) is included.
    for end_month in range(min_obs, n + 1, step):
        # rolling window starts min_obs rows back, expanding window always starts at row 0
        first_month = end_month - min_obs if max_obs else 0

        data_slice = excess_return_df.filter(
            (pl.col("Order") >= first_month) & (pl.col("Order") < end_month)
        )

        ticker_list = list(data_slice.select(pl.all().exclude("Order","Date")).columns)

        date = data_slice["Date"][-1]

        ret = data_slice.select(pl.all().exclude("Date", "Order")).drop_nulls().to_numpy()

        # LedoitWolf needs a full window of months where EVERY ticker has a return.
        # Shorter windows (e.g. before a stock existed) get no Sigma (null).
        if ret.shape[0] >= min_obs:

            lw = LedoitWolf().fit(ret)
            Sigma = lw.covariance_
            Sigma = Sigma.tolist()
            Mu = ret.mean(axis=0).tolist()
        else:
            Sigma = None
            Mu = None
        # number of months in this window (T): Michaud resampling simulates fake histories of the same length
        obs_num = ret.shape[0]

        results.append({
            "Date": date,
            "VCV": Sigma,
            "MU":Mu,
            "Tickers": ticker_list,
            "Obs_num":obs_num
        })
    


    df = pl.DataFrame(results, schema=schema)
    #Example structure :
    """
┌────────────┬─────────────────────────────────┬──────────────────┐
│ Date       ┆ VCV                             ┆ Tickers          │
│ ---        ┆ ---                             ┆ ---              │
│ date       ┆ list[list[f64]]                 ┆ list[str]        │
╞════════════╪═════════════════════════════════╪══════════════════╡
│ 1996-12-31 ┆ [[0.002479, 0.000353], [0.0003… ┆ ["MSFT", "AAPL"] │
    """

    return df

def Create_excess_return_df(tickers:list):
    #Load the DataFrame
    risk_free_rate = pl.read_parquet(MONTHLY_DATA_PATH/"^FVX.parquet")
    #Keep only the interesting columns
    risk_free_rate = risk_free_rate.select("Date","Close")
    #Rescale
    risk_free_rate = risk_free_rate.with_columns((pl.col("Close")/100/12).alias("RF"))
    df = risk_free_rate.select(pl.all().exclude("Close"))

    for ticker in tickers:
        stock_df = pl.read_parquet(MONTHLY_DATA_PATH/f"{ticker}.parquet")
        close_df = stock_df.select("Date","Close")
        #compute returns
        close_df = close_df.with_columns(pl.col("Close").pct_change().alias(f"{ticker}"))
        close_df = close_df.select("Date",f"{ticker}")
        df = df.join(close_df, on="Date", how="left")

        #compute Excess return
        df = df.with_columns(pl.col(f"{ticker}")-pl.col("RF").alias(f"{ticker}"))

    result = df.select(pl.all().exclude("RF"))

    return result




def Summary_perf_df(weights_df, sigma_df, name_of_weight ="Weights"):



    #join weights at t+1
    weights_df = weights_df.with_columns(pl.col(name_of_weight).shift(1))
    result_df = sigma_df.join(weights_df, on="Date", how="left")


    #Save the computed port returns and volatilites
    perf_port = []


    for result in result_df.iter_rows(named=True):

        VCV = result["VCV"]
        MU = result["MU"]
        weight = result["Weights"]
        date = result["Date"]


        if VCV is not None and MU is not None and weight is not None:

            VCV = np.array(VCV, dtype=float)
            MU = np.array(MU, dtype=float)
            weight = np.array(weight, dtype=float)

            port_ret = MU @ weight.T
            port_vol = weight @ VCV @ weight.T

        else:
            port_ret = None
            port_vol = None

        perf_port.append({
            "Date": date,
            "VCV": VCV,
            "MU": MU,
            "weight": weight,
            "port_ret": port_ret,
            "port_vol": np.sqrt(port_vol) if port_vol is not None else None,  # np.sqrt(None) would crash on dates without weights
            "port_ret_annual":port_ret * 12 if port_ret is not None else None,
            "port_vol_annual":np.sqrt(port_vol) * np.sqrt(12) if port_vol is not None else None,
            "port_sharpe": port_ret*12 /(np.sqrt(port_vol) * np.sqrt(12)) if port_ret is not None and port_vol is not None else None,
        })
    perf_port_df = pl.DataFrame(perf_port, schema={"Date": pl.Date,
                                                   "VCV": pl.List(pl.List(pl.Float64)),
                                                   "MU": pl.List(pl.Float64),
                                                   "weight": pl.List(pl.Float64),
                                                   "port_ret": pl.Float64,
                                                   "port_vol": pl.Float64,
                                                   "port_ret_annual": pl.Float64,
                                                   "port_vol_annual": pl.Float64,
                                                   "port_sharpe": pl.Float64

                                                   })
    return perf_port_df
















