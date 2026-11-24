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
    universe = [c for c in excess_return_df.columns if c not in ("Order", "Date") and not c.endswith("_RET")]

    results = []
    schema = pl.Schema({
        "Date": pl.Date,
        "VCV": pl.List(pl.List(pl.Float64)),
        "MU": pl.List(pl.Float64),  # one mean excess return per ticker (a flat list, not a matrix)
        "MU_TOTAL": pl.List(pl.Float64),  # one mean full return per ticker (excess + RF), same order as MU
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

        counts = dict(zip(universe, data_slice.select(pl.col(universe).count()).row(0)))
        ticker_list = [t for t in universe if counts[t] >= min_obs]

        date = data_slice["Date"][-1]

        while True:
            total_list = [f"{t}_RET" for t in ticker_list]
            clean = data_slice.select(ticker_list + total_list).drop_nulls()
            if clean.height >= min_obs or not ticker_list:
                break
            ticker_list.remove(min(ticker_list, key=counts.get))
        ret = clean.select(ticker_list).to_numpy()
        ret_total = clean.select(total_list).to_numpy()

        # LedoitWolf needs a full window of months where EVERY ticker has a return.
        # Shorter windows (e.g. before a stock existed) get no Sigma (null).
        if ticker_list and ret.shape[0] >= min_obs:

            lw = LedoitWolf().fit(ret)
            Sigma = lw.covariance_
            Sigma = Sigma.tolist()
            Mu = ret.mean(axis=0).tolist()
            Mu_total = ret_total.mean(axis=0).tolist()
        else:
            Sigma = None
            Mu = None
            Mu_total = None
        # number of months in this window (T): Michaud resampling simulates fake histories of the same length
        obs_num = ret.shape[0]

        results.append({
            "Date": date,
            "VCV": Sigma,
            "MU":Mu,
            "MU_TOTAL":Mu_total,
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

#New helper to create the excess return with the cleaned data
def Create_excess_return_df(tickers: list):
    master_df = pl.read_parquet(MASTER_DATA_PATH/"Master.parquet")
    df = master_df.select("Date", "RF_CHF", *[pl.col(t).alias(f"{t}_RET") for t in tickers])  # full returns
    df = df.with_columns((pl.col(f"{t}_RET") - pl.col("RF_CHF")).alias(t) for t in tickers)   # excess
    return df.select(pl.all().exclude("RF_CHF")).sort("Date")


#Old function depreciated with new data format.
def Create_excess_return_df_depr(tickers:list):
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
        close_df = close_df.with_columns(pl.col("Close").pct_change().alias(f"{ticker}_RET"))
        close_df = close_df.select("Date",f"{ticker}_RET")
        df = df.join(close_df, on="Date", how="left")

        #compute Excess return (parentheses: the alias must name the difference, not RF)
        df = df.with_columns((pl.col(f"{ticker}_RET")-pl.col("RF")).alias(f"{ticker}"))

    result = df.select(pl.all().exclude("RF"))

    return result




def Summary_perf_df(weights_df, sigma_df, tickers, name_of_weight ="Weights"):
    pos_of = {t: i for i, t in enumerate(tickers)}



    #join weights at t+1
    weights_df = weights_df.with_columns(pl.col(name_of_weight).shift(1))
    result_df = sigma_df.join(weights_df, on="Date", how="left")


    #Save the computed port returns and volatilites
    perf_port = []


    for result in result_df.iter_rows(named=True):

        VCV = result["VCV"]
        MU = result["MU"]
        MU_TOTAL = result["MU_TOTAL"]
        weight = result[name_of_weight]
        date = result["Date"]


        if VCV is not None and MU is not None and weight is not None:

            VCV = np.array(VCV, dtype=float)
            MU = np.array(MU, dtype=float)
            MU_TOT = np.array(MU_TOTAL, dtype=float)
            weight = np.array(weight, dtype=float)
            w = weight[[pos_of[t] for t in result["Tickers"]]]

            port_ret = MU @ w
            port_ret_tot = MU_TOT @ w
            port_vol = w @ VCV @ w

        else:
            port_ret = None
            port_ret_tot = None
            port_vol = None

        perf_port.append({
            "Date": date,
            "VCV": VCV,
            "MU": MU,
            "weight": weight,
            "port_ret": port_ret,
            "port_ret_tot": port_ret_tot,
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
                                                   "port_ret_tot": pl.Float64,
                                                   "port_vol": pl.Float64,
                                                   "port_ret_annual": pl.Float64,
                                                   "port_vol_annual": pl.Float64,
                                                   "port_sharpe": pl.Float64

                                                   })
    return perf_port_df



if __name__ == "__main__":
    pass












