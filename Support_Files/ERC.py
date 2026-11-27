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
import pandas as pd

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
def ERC_optimization(ticker_list:list, start_date, end_date, step = 1, min_obs = 24, max_obs:bool = False, constraints = None):
    #a default dict in the signature is built once at import and shared by every call, so build it here
    if constraints is None:
        constraints = new_info_constraints()
    excess_df = Create_excess_return_df(ticker_list)
    sigma_df = rolling_sigmas(excess_return_df=excess_df,start_date= start_date, end_date=end_date, step=step, max_obs=max_obs,min_obs= min_obs)
    N = len(ticker_list)

    rows = []
    for row in sigma_df.iter_rows(named=True):
        VCV = row["VCV"]
        if VCV is None:
            rows.append({"Date": row["Date"], "Weights": None})   # not enough data for this date
            continue
        else:
            # the months this VCV was estimated from (same rules as rolling_sigmas)
            tickers = row["Tickers"]
            window = (excess_df
                      .filter((pl.col("Date") >= start_date) & (pl.col("Date") <= row["Date"]))
                      .sort("Date")
                      .select("Date", *tickers)
                      .drop_nulls())                       # only months where every ticker has a return
            if max_obs:                                    # rolling window: only the last min_obs months
                window = window.tail(min_obs)
            returns = window.select(tickers).to_numpy()    # shape (months, tickers), same order as VCV


            #ERC caps: None = no cap, a number = same cap for every ticker,
            #a dict {ticker: cap} = own cap per ticker (tickers not in the dict: no cap)
            erc_caps = constraints.get("ERC_Cap")
            if erc_caps is None:
                max_weights = np.ones(len(tickers))
            elif isinstance(erc_caps, dict):
                max_weights = np.array([erc_caps.get(t, 1.0) for t in tickers], dtype=float)
            else:
                max_weights = np.full(len(tickers), float(erc_caps))
            #the weights must add up to 100 %: caps summing to less are infeasible -> no ERC weights on this date
            if max_weights.sum() < 1 - 1e-9:
                rows.append({"Date": row["Date"], "Weights": None})
                continue

            #ERC risk budget: None = every asset gets the same share of risk,
            #otherwise each class gets its share, split equally between the class's tradable assets
            risk_budget = None
            group_cfg = constraints.get("ERC_Group_Budget")
            if group_cfg is not None:
                groups = [group_cfg["groups"][t] for t in tickers]
                risk_budget = np.array([group_cfg["budget"][g] / groups.count(g) for g in groups])
                risk_budget = risk_budget / risk_budget.sum()     #shares of risk add up to 1

            model = RiskBudgeting(prior_estimator=CustomPrior(np.array(VCV, dtype=float)),
                                  max_weights=max_weights, risk_budget=risk_budget)
            model.fit(returns)
            weight = np.zeros(N)
            weight[[ticker_list.index(t) for t in tickers]] = model.weights_
            rows.append({"Date": row["Date"], "Weights": weight.tolist()})

    weights_df = pl.DataFrame(rows, schema={"Date": pl.Date, "Weights": pl.List(pl.Float64)})
    return {"weights_df": weights_df, "sigma_df": sigma_df}

if __name__ == "__main__":
    constraints = new_info_constraints()
    constraints = ERC_add_max_weight_per_ticker(constraints, tickers=["Gold"], max_weight=[0.01])
    print(ERC_optimization(ticker_list=TICKER_LIST, start_date=pl.date(year=1970, month=1, day=1),
                           end_date=pl.date(year=2025, month=12, day=1), step=1, max_obs=False, min_obs=24,constraints=constraints))
    ERC_results = ERC_optimization(ticker_list=["Gold","Silver"], start_date=pl.date(year=1970, month=1, day=1),
                           end_date=pl.date(year=2025, month=12, day=1), step=1, max_obs=False, min_obs=24, constraints=constraints)
    print(ERC_results)