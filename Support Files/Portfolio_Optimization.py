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

#All code that is used for normal constrained optimization

def rolling_weights(ticker_list:list,constraints, start_date, end_date, step = 1, min_obs = 24, max_obs:bool = False):
    excess_returns = Create_excess_return_df(ticker_list)
    sigma_df = rolling_sigmas(excess_returns, start_date = start_date, end_date = end_date, step = step, min_obs = min_obs, max_obs = max_obs)
    # no window at all -> the chosen period is shorter than min_obs months
    if sigma_df.height == 0:
        raise ValueError(f"The period between start_date and end_date has fewer than min_obs={min_obs} months.")
    n = len(sigma_df["Tickers"][0])
    ones = np.ones(n)

    #which portfolio we look for: "MVP" (min variance) or "Tangency" (max Sharpe), read like Long_only
    objective = constraints.get("Objective", "MVP")
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"Objective must be one of {VALID_OBJECTIVES}, got {objective!r}")
    tangency = objective == "Tangency"

    # Two sets of rows:
    #  cons_w : rows on the real weights w (sum(w) = 1). Only used to check which returns are reachable.
    #  cons   : rows given to the solver. Same as cons_w for the MVP. For the tangency the solver works on
    #           y = scaled weights (mu @ y = 1 instead of sum = 1), so some rows are written differently.
    cons_w = new_constraint_dict()
    budget(cons_w, n)

    cons = new_constraint_dict()
    if not tangency:
        #compulsory to have sum of assets to 1 for the MVP
        budget(cons, n)
    else:
        #tangency: no budget, the scale is set per date by mu @ y = 1 (in the loop); sum(y) must stay positive
        add_inequality(-ones, 0.0, cons)

    #checks whether we are limited to long only transactions, triggers when True
    if constraints["Long_only"]:
        long_only(cons_w, n)
        long_only(cons, n)  # same row for w and y: dividing by a positive number keeps the sign

    #checks whether there are limitation of maximal weight given to one asset
    if constraints["Max_Weight"] is not None:
        caps = per_asset(constraints["Max_Weight"], ticker_list, default=1.0)
        if caps.sum() < 1:  # replaces cap * n < 1
            raise ValueError(f"Caps sum to {caps.sum():.2f} < 1: weights can't reach 100%")
        max_weight(cons_w, n, caps)
        if not tangency:
            max_weight(cons, n, caps)  # w_i <= cap_i
        else:
            # w_i <= cap_i with w = y / sum(y)  ->  y_i - cap_i * sum(y) <= 0
            add_inequality(np.eye(n) - np.outer(caps, ones), np.zeros(n), cons)

    #checks whether there are any relative weights to respect, to be implemented
    if constraints["Relative_Weight"] is not None:
        pass

    #checks whether there is a target return: only read the setting here,
    #the row itself depends on mu, which changes every date -> added in the loop
    target_cfg = constraints["Target_Ret"]

    rows = []
    for row in sigma_df.iter_rows(named=True):
        weight = None
        # Check the raw value first: np.array(None, dtype=float) gives array(nan), which is not None
        if row["VCV"] is not None:  # None = not enough data for this date
            sigma = np.array(row["VCV"], dtype=float)
            mu = np.array(row["MU"], dtype=float)  # this date's mu, one per ticker
            cons_t = dict(cons)      # copies: this date's rows must not pile up in cons / cons_w
            cons_w_t = dict(cons_w)
            reachable = True

            #Satisfies the target return needed
            if target_cfg is not None:
                exact = target_cfg["kind"] == "exact"
                target = target_cfg["target"]
                # lowest / highest return the constraints allow on this date (with or without long-only)
                lo, hi = return_range(mu, cons_w)
                if target > hi or (exact and target < lo):
                    reachable = False  # unreachable this date
                else:
                    target_return(cons_w_t, mu, target, exact)  # w-form, used by the tangency check below
                    if not tangency:
                        target_return(cons_t, mu, target, exact)         # mu @ w >= target (or ==)
                    elif exact:
                        add_equality(mu - target * ones, 0.0, cons_t)    # (mu - target) @ y == 0
                    else:
                        add_inequality(target * ones - mu, 0.0, cons_t)  # (target - mu) @ y <= 0

            #Tangency: fix the scale of y with this date's mu
            if reachable and tangency:
                # needs at least one allowed portfolio with a positive expected return, otherwise mu @ y = 1 is impossible
                if return_range(mu, cons_w_t)[1] <= 0:
                    reachable = False
                else:
                    add_equality(mu, 1.0, cons_t)  # mu @ y = 1

            if reachable:
                weight = Optimize_weights(sigma, cons_t)
                #Tangency: the solver returned y -> rescale to weights that sum to 1
                if tangency and weight is not None:
                    weight = weight / weight.sum() if weight.sum() > 1e-12 else None

        rows.append({"Date": row["Date"], "Weights": None if weight is None else weight.tolist()})

    weights_df = pl.DataFrame(rows, schema={"Date": pl.Date, "Weights": pl.List(pl.Float64)})
    print(weights_df)
    return {"weights_df": weights_df, "sigma_df": sigma_df, "objective": objective}

def Optimize_weights(Sigma, constraint_dict=None):
    if constraint_dict is None:
        constraint_dict = new_constraint_dict()
    n = Sigma.shape[0]

    P = matrix(np.asarray(Sigma, dtype=float))
    q = matrix(np.zeros(n))

    G = to_cvx(constraint_dict["G"])
    h = to_cvx(constraint_dict["h"])
    A = to_cvx(constraint_dict["A"])
    b = to_cvx(constraint_dict["b"])

    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except ValueError:  # e.g. "domain error" when no portfolio satisfies the rows
        return None
    if sol["status"] != "optimal":
        print("Optimization failed")
        return None
    return np.array(sol["x"]).ravel()

if __name__ == "__main__":

    constraints = new_info_constraints()


    constraints = add_long_only_click(constraints)

    constraints = set_objective(constraints, "Tangency")
    print(constraints)



    result = rolling_weights(ticker_list=["MSFT","AAPL","^RUT"],constraints=constraints,start_date=pl.date(year=1970,month=1,day=1),end_date=pl.date(year=2025,month=12,day=1),step=1,max_obs=False,min_obs=24)
    summary = summary_MVP_df(result["weights_df"], result["sigma_df"])
    print(summary)
