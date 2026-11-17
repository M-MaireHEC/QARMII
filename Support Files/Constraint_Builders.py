import numpy as np
from cvxopt import matrix, solvers
from typing import Final
import polars as pl
import sklearn
from sklearn.covariance import LedoitWolf
from Config import *
from skfolio.optimization import RiskBudgeting
from skfolio.prior import BasePrior, ReturnDistribution
from Config import *
from Tools import *


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

def to_cvx(x):
    return None if x is None else matrix(np.asarray(x, dtype=float))

#Lowest and highest portfolio return the rows in cd allow (two small linear programs)
def return_range(mu, cd):
    if cd["G"] is None:  # only the budget: unlimited shorting -> any return is reachable
        return -np.inf, np.inf
    c = matrix(np.asarray(mu, float))
    G, h, A, b = (to_cvx(cd[k]) for k in ("G", "h", "A", "b"))
    lo = solvers.lp(c, G, h, A, b)   # minimize  mu @ w
    hi = solvers.lp(-c, G, h, A, b)  # maximize  mu @ w
    lo_v = lo["primal objective"] if lo["status"] == "optimal" else -np.inf
    hi_v = -hi["primal objective"] if hi["status"] == "optimal" else np.inf
    return lo_v, hi_v



def target_return(cd, mu, target, exact=True):
    """exact: mu @ w == target   |   minimum: mu @ w >= target"""
    mu = np.asarray(mu, float)
    if exact:
        return add_equality(mu, target, cd)          # goes next to the budget row in A / b
    return add_inequality(-mu, -target, cd)          # -mu @ w <= -target  <=>  mu @ w >= target


VALID_OBJECTIVES = ("MVP", "Tangency")

#Build the information to be passed to rolling weights
def new_info_constraints():
    return {
        "Long_only":False,
        "Max_Weight":None,
        "Relative_Weight":None,
        "Target_Ret":None,
        "Objective":"MVP",  # "MVP" = min variance, "Tangency" = max Sharpe
    }

def add_long_only_click(constraints):
    #Flip flop True to False , False to True on click
    if constraints["Long_only"]:
        constraints["Long_only"] = False
    else:
        constraints["Long_only"] = True
    return constraints

def add_max_weight_per_ticker(constraints, tickers:list, max_weight:list):
    dic_ticker_weight = {}
    for ticker in tickers:
        position = tickers.index(ticker)
        dic_ticker_weight[ticker] = max_weight[position]
    constraints["Max_Weight"] = dic_ticker_weight
    return constraints

def add_target_ret(constraints, kind: str, target: float):
    # kind: "min" (portfolio return >= target) or "exact" (== target)
    # target: monthly excess return, e.g. 0.10 / 12 for 10% a year
    constraints["Target_Ret"] = {"kind": kind, "target": target}
    return constraints

def set_objective(constraints, objective: str):
    # "MVP" (min variance) or "Tangency" (max Sharpe)
    if objective not in VALID_OBJECTIVES:  # catches typos like "MaxSharpe"
        raise ValueError(f"Objective must be one of {VALID_OBJECTIVES}, got {objective!r}")
    constraints["Objective"] = objective
    return constraints

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

#Only long position
def long_only(cd, n):
    return add_inequality(-np.eye(n), np.zeros(n), cd)

#sums to 1
def budget(cd, n):
    return add_equality(np.ones(n), 1.0, cd)

#Set max weight of ALL
def max_weight(cd, n, cap):
    return add_inequality(np.eye(n), np.full(n, cap), cd)

def per_asset(value, tickers, default):
    """Number -> same value for every asset. Dict -> one value per ticker, others get `default`."""
    if isinstance(value, dict):
        unknown = set(value) - set(tickers)
        if unknown:                                    # catches typos like "APPL"
            raise ValueError(f"Unknown tickers in constraint: {sorted(unknown)}")
        return np.array([value.get(t, default) for t in tickers], dtype=float)
    return np.full(len(tickers), value, dtype=float)