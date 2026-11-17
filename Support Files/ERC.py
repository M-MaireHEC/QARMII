import numpy as np
from cvxopt import matrix, solvers
from typing import Final
import polars as pl
import sklearn
from sklearn.covariance import LedoitWolf
from Config import *
from skfolio.optimization import RiskBudgeting
from skfolio.prior import BasePrior, ReturnDistribution
from Tools import *
from Constraint_Builders import *

#ALL code common to ERC



#SKfolio needs a change to accept Covariance estimates to be passed in the function
class CustomPrior(BasePrior):

    def __init__(self, covariance):
        self.covariance = covariance

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)                     # works for numpy, pandas and polars input
        self.return_distribution_ = ReturnDistribution(
            mu=X.mean(axis=0),                             # same as your MU column, instead of zeros
            covariance=np.asarray(self.covariance, dtype=float),
            returns=X,
        )
        return self


#ERC Optimization
def ERC_optimization(ticker_list:list, start_date, end_date, step = 1, min_obs = 24, max_obs:bool = False):
    excess_df = Create_excess_return_df(ticker_list)
    sigma_df = rolling_sigmas(excess_return_df=excess_df,start_date= start_date, end_date=end_date, step=step, max_obs=max_obs,min_obs= min_obs)
    tickers = sigma_df["Tickers"][0].to_list()            # column order of every VCV

    rows = []
    for row in sigma_df.iter_rows(named=True):
        VCV = row["VCV"]
        if VCV is None:
            rows.append({"Date": row["Date"], "Weights": None})   # not enough data for this date
            continue
        else:
            # the months this VCV was estimated from (same rules as rolling_sigmas)
            window = (excess_df
                      .filter((pl.col("Date") >= start_date) & (pl.col("Date") <= row["Date"]))
                      .sort("Date")
                      .drop_nulls())                       # only months where every ticker has a return
            if max_obs:                                    # rolling window: only the last min_obs months
                window = window.tail(min_obs)
            returns = window.select(tickers).to_numpy()    # shape (months, tickers), same order as VCV

            model = RiskBudgeting(prior_estimator=CustomPrior(np.array(VCV, dtype=float)))
            model.fit(returns)
            rows.append({"Date": row["Date"], "Weights": model.weights_.tolist()})

    weights_df = pl.DataFrame(rows, schema={"Date": pl.Date, "Weights": pl.List(pl.Float64)})
    return {"weights_df": weights_df, "sigma_df": sigma_df}

if __name__ == "__main__":
    print(ERC_optimization(ticker_list=["MSFT", "AAPL", "^RUT"], start_date=pl.date(year=1970, month=1, day=1),
                           end_date=pl.date(year=2025, month=12, day=1), step=1, max_obs=False, min_obs=24))