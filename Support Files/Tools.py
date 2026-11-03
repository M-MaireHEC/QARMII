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

#Returns the optimal weights for the constrained problem
def Optimize_weights(Sigma, constraint_dict = cd):
    """
    Loss function: min 1/2 x.T @ P @ x + q.T @ x, with G @ x <= h , A @ x = b as given by CVXOPT
    """
    #Define the P matrix
    P = matrix(Sigma)

    #Define a q vector full of 0. We do not need of this part as we will pass the required return through a constraint. By defining q we are mean-variance optimizing
    q = matrix(np.zeros(Sigma.shape[0]))

    A = matrix(constraint_dict.get('A', None))
    b = matrix(constraint_dict.get('b', None))

    G = matrix(constraint_dict.get("G", None))
    h = matrix(constraint_dict.get('h', None))

    sol = solvers.qp(P, q, G, h, A, b)

    #Return None if the solution is non feasable :
    if sol["status"] != "optimal":
        print("Optimization failed")
        return None
    return sol["x"]

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

def long_only(cd = cd, n = specs["rows"]):
    # w >= 0  ⇔  -I @ w <= 0
    return add_inequality(cd, -np.eye(n), np.zeros(n))

def budget(cd = cd, n = specs["rows"]):
    # sum(w) == 1
    return add_equality(cd, np.ones(n), 1.0)

def max_weight(cap, cd = cd, n= specs["rows"]):
    # w_i <= cap for every asset
    return add_inequality(cd, np.eye(n), np.full(n, cap))

#Preparing the Sigmas per period.
def rolling_sigmas(excess_return_df:pl.DataFrame, start_date, end_date, step = 1, min_obs = 12, max_obs = None):
    #max_obs = 1 means the number of observation increase. Else limits the number of observations
    excess_return_df = excess_return_df.filter(
        (pl.col("Date")>=start_date) & (pl.col("Date")<=end_date)
    )
    n = excess_return_df.shape[0]
    first_cum_month = excess_return_df["Cum_Month"][0]
    end_cum_month = first_cum_month+min_obs

    data_slice = excess_return_df.filter(
        (pl.col("Cum_Month")>= first_cum_month) & (pl.col("Cum_Month") <= end_cum_month)
    )

    ret_df = data_slice.select(pl.all().exclude("Date","Month","Cum_Month"))
    Sigma = LedoitWolf(ret_df)
    print(Sigma)

#Tests
if __name__ == "__main__":
    df = pl.read_parquet(MONTHLY_DATA_PATH/"AAPL.parquet")
    print(df)








