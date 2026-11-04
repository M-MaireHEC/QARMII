import numpy as np
from cvxopt import matrix, solvers
from typing import Final
import polars as pl
import sklearn
from sklearn.covariance import LedoitWolf
from Config import *

def new_constraint_dict():
    return {"A": None, "b": None, "G": None, "h": None}

cd = new_constraint_dict()

#Get common needed information for the
def Get_specs(Sigma):
    rows = Sigma.shape[0]
    cols = Sigma.shape[1]

    return {"rows":rows, "cols":cols}

specs:Final = {
    "rows":1
}

def _to_cvx(x):
    return None if x is None else matrix(np.asarray(x, dtype=float))

def Optimize_weights(Sigma, constraint_dict=None):
    if constraint_dict is None:
        constraint_dict = new_constraint_dict()
    n = Sigma.shape[0]

    P = matrix(np.asarray(Sigma, dtype=float))
    q = matrix(np.zeros(n))

    G = _to_cvx(constraint_dict["G"])
    h = _to_cvx(constraint_dict["h"])
    A = _to_cvx(constraint_dict["A"])
    b = _to_cvx(constraint_dict["b"])

    sol = solvers.qp(P, q, G, h, A, b)
    if sol["status"] != "optimal":
        print("Optimization failed")
        return None
    return np.array(sol["x"]).ravel()

def add_inequality( G_new, h_new, cd = cd):
    """Append rows  G_new @ w <= h_new  to the constraint dict."""
    G_new = np.atleast_2d(np.asarray(G_new, float))   # a single row becomes shape (1, n)
    h_new = np.atleast_1d(np.asarray(h_new, float))   # a scalar becomes shape (1,)

    if cd["G"] is None:
        cd["G"], cd["h"] = G_new, h_new
    else:
        cd["G"] = np.vstack([cd["G"], G_new])
        cd["h"] = np.concatenate([cd["h"], h_new])
    return cd


def add_equality( A_new, b_new, cd = cd):
    """Append rows  A_new @ w == b_new  to the constraint dict."""
    A_new = np.atleast_2d(np.asarray(A_new, float))
    b_new = np.atleast_1d(np.asarray(b_new, float))

    if cd["A"] is None:
        cd["A"], cd["b"] = A_new, b_new
    else:
        cd["A"] = np.vstack([cd["A"], A_new])
        cd["b"] = np.concatenate([cd["b"], b_new])
    return cd

def long_only(cd, n):
    return add_inequality(-np.eye(n), np.zeros(n), cd)

def budget(cd, n):
    return add_equality(np.ones(n), 1.0, cd)

def max_weight(cd, n, cap):
    return add_inequality(np.eye(n), np.full(n, cap), cd)

#Preparing the Sigmas per period (method uses LedoitWolf).
def rolling_sigmas(excess_return_df:pl.DataFrame, start_date, end_date, step = 1, min_obs = 12, max_obs:bool = False):
    #max_obs = 1 means the number of observation increase. Else limits the number of observations
    excess_return_df = excess_return_df.filter(
        (pl.col("Date")>=start_date) & (pl.col("Date")<=end_date)
    )

    excess_return_df = excess_return_df.sort("Date").with_row_index("Order")

    n = excess_return_df.shape[0]

    first_month = 0
    end_month = first_month + min_obs

    results = []
    schema = pl.Schema({
        "Date": pl.Date,
        "VCV": pl.List(pl.List(pl.Float64)),
        "Tickers": pl.List(str)
    })

    for i in range(n-min_obs):

        data_slice = excess_return_df.filter(
            (pl.col("Order") >= first_month) & (pl.col("Order") < end_month)
        )

        ticker_list = list(data_slice.select(pl.all().exclude("Order","Date")).columns)

        date = data_slice["Date"][-1]

        ret = data_slice.select(pl.all().exclude("Date", "Order")).drop_nulls().to_numpy()
        lw = LedoitWolf().fit(ret)
        Sigma = lw.covariance_


        results.append({
            "Date": date,
            "VCV": Sigma.tolist(),
            "Tickers": ticker_list
        })
    

        if max_obs :
            first_month += step

        end_month += step

    df = pl.DataFrame(results, schema=schema)
    #Example structure :
    """
    ────────────┬─────────────────────────────────┐
    │ Date       ┆ VCV                             │
    │ ---        ┆ ---                             │
    │ date       ┆ list[list[f64]]                 │
    ╞════════════╪═════════════════════════════════╡
    │ 1996 - 12 - 31 ┆ [[0.002704, 0.000304], [0.0003… │
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
        df = df.with_columns(pl.col(f"{ticker}")-pl.col("RF"))

    result = df.select(pl.all().exclude("RF"))

    return result




#Tests
if __name__ == "__main__":

    df = Create_excess_return_df(["MSFT","AAPL"])
    Sigma_df = rolling_sigmas(df,start_date = pl.date(1996,1,1), end_date = pl.date(2020,1,1), step = 1)
    print(Sigma_df)
    element = Sigma_df["VCV"][0]

    Sigma = np.array(Sigma_df["VCV"][0].to_list(), dtype=np.float64)
    n = Sigma.shape[0]

    cons = new_constraint_dict()
    budget(cons, n)
    long_only(cons, n)

    w = Optimize_weights(Sigma, cons)
    print(w)







