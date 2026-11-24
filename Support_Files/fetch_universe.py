"""
fetch_universe.py  -  QARM II data set: monthly returns in CHF, one source per instrument.

Inputs  (<project>/Data_Input/):
  fred/*.csv           FRED: FX rates (DEXSZUS, DEXUSUK, DEXUSEU, EXSZUS, EXGEUS) and
                       CHF 3-month rates (IR3TIB01CHM156N, IR3TED01CHM156N)
  investing/*.csv      investing.com monthly files: gold, silver, Thomson Reuters government
                       bond benchmark total-return indices, SPI, CAC 40 Gross TR, FTSE All-Share TR
  msci/*.xls           MSCI USA Small Cap, Gross (total return), USD, monthly (msci.com)
Outputs (<project>/Data_Output/universe/):
  assets_returns_chf_monthly.csv      asset classes
  smi_stocks_returns_chf_monthly.csv  Swiss stocks (current + former SMI members)
  rf_chf_monthly.csv                  CHF risk-free return per month
"""
import time
from pathlib import Path
from Config import *
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent          # script sits in "Support_Files"
IN, OUT = ROOT / "Data_Input", ROOT / "Data_Output" / "universe"
OUT.mkdir(parents=True, exist_ok=True)

START = "1990-01-01"
HISTORY_MONTHS = MIN_OBS     # months of data an asset needs before it can be traded (10 years)
END = pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)   # last complete month
try:
    pd.date_range("2000-01-31", periods=1, freq="ME")
    ME = "ME"                                                          # pandas >= 2.2
except ValueError:
    ME = "M"

# name: (source, id, currency)
#   investing: monthly file in Data_Input/investing (index levels, already total return)
#   yahoo:     index / mutual fund / stock; total return from Yahoo close + Yahoo dividends
#   msci:      MSCI index (Gross = dividends reinvested, USD), found by its column name
# "TR ... Government Benchmark" = Thomson Reuters total-return index of the benchmark bond
# of that maturity (coupons reinvested).
                       # unadjusted split on Yahoo


def fred(sid):
    """FRED series from any CSV in Data_Input/fred, as month-end values."""
    for f in (IN / "fred").glob("*.csv"):
        df = pd.read_csv(f, index_col=0, parse_dates=True, na_values=".")
        if sid in df.columns:
            return pd.to_numeric(df[sid], errors="coerce").dropna().resample(ME).last()
    raise FileNotFoundError(f"FRED series {sid} not found in {IN / 'fred'}")


def monthly(px):
    """Month-end prices -> monthly returns up to the last complete month,
    dropping leading months with zero return (stale placeholder prices)."""
    m = px.resample(ME).last()
    r = m[(m.index >= START) & (m.index <= END)].pct_change(fill_method=None).dropna()
    return r.loc[r.ne(0).idxmax():]


def yahoo_tr(ticker):
    """Daily total-return index built from raw close + dividends (Yahoo's Adj Close is
    unreliable for some listings). Dividends above 25% of the price are data errors."""
    tk = yf.Ticker(ticker)
    h = tk.history(start=START, auto_adjust=False)
    time.sleep(0.3)
    if h.empty:
        return None
    h.index = h.index.tz_localize(None).normalize()
    h = h[~h.index.duplicated(keep="last")]
    close, div = h["Close"], h["Dividends"].fillna(0.0)
    if tk.history_metadata.get("currency") in ("GBp", "GBX"):     # LSE: price in pence
        paid = div[div > 0]
        if len(paid) and (paid / close.shift(1)[paid.index]).median() < 0.001:
            div = div * 100                                        # dividends given in GBP
    div = div.where(div / close.shift(1) <= 0.25, 0.0)
    return ((close + div) / close.shift(1)).fillna(1.0).cumprod()


def investing(fname):
    """investing.com monthly file: 'Date' = MM/01/YYYY, 'Price' = that month's close."""
    df = pd.read_csv(IN / "investing" / fname, thousands=",")
    return pd.Series(df["Price"].values,
                     index=pd.to_datetime(df["Date"], format="%m/%d/%Y")).sort_index()


def msci(prefix):
    """MSCI 'historyIndex.xls' downloads in Data_Input/msci (any file name, one or more indices
    per file). Only files with 'Index Level : Gross' and 'Currency : USD' are used."""
    for f in sorted((IN / "msci").glob("*.xls*")):
        raw = pd.read_excel(f, header=None)
        meta = {str(raw.iat[i, 0]).strip(): str(raw.iat[i, 1]).strip() for i in range(min(10, len(raw)))}
        if meta.get("Index Level :") != "Gross" or meta.get("Currency :") != "USD":
            continue
        hdr = raw.index[raw[0].astype(str).str.strip() == "Date"][0]
        for j in range(1, raw.shape[1]):
            if str(raw.iat[hdr, j]).upper().startswith(prefix.upper()):
                body = raw.iloc[hdr + 1:, [0, j]]
                d = pd.to_datetime(body[0], format="%b %d, %Y", errors="coerce")
                v = pd.to_numeric(body[j].astype(str).str.replace(",", ""), errors="coerce")
                px = pd.Series(v.values, index=d).dropna()
                return px[px.index.notna()].sort_index()
    return None


def check_msci_files():
    """Report every MSCI file and warn about the ones that will be ignored."""
    for f in sorted((IN / "msci").glob("*.xls*")):
        raw = pd.read_excel(f, header=None)
        meta = {str(raw.iat[i, 0]).strip(): str(raw.iat[i, 1]).strip() for i in range(min(10, len(raw)))}
        hdr = raw.index[raw[0].astype(str).str.strip() == "Date"]
        names = [str(x) for x in raw.iloc[hdr[0], 1:].dropna()] if len(hdr) else []
        ok = meta.get("Index Level :") == "Gross" and meta.get("Currency :") == "USD"
        print(f"{'OK' if ok else '!! IGNORED'} {f.name}: level={meta.get('Index Level :')}, "
              f"currency={meta.get('Currency :')}, indices={names}")


def fx_chf():
    """Month-end CHF per 1 USD / GBP / EUR. Before 1999: EUR via DEM (1.95583 DEM per EUR),
    using monthly averages on both legs because DEM/USD only exists as averages."""
    usd = fred("DEXSZUS")
    eur = (fred("DEXUSEU") * usd).dropna()
    eur_pre = fred("EXSZUS") / fred("EXGEUS") * 1.95583
    eur = eur.combine_first(eur_pre[eur_pre.index < "1999-01-31"])
    return pd.DataFrame({"USD": usd, "GBP": fred("DEXUSUK") * usd, "EUR": eur})


def rf_chf():
    """3-month CHF interbank rate; before mid-1999 the 3-month CHF euro-deposit rate.
    The rate observed in month t-1 is earned in month t."""
    r = fred("IR3TIB01CHM156N").combine_first(fred("IR3TED01CHM156N"))
    rf = ((1 + r / 100) ** (1 / 12) - 1).shift(1)
    rf = rf.reindex(pd.date_range(rf.index[0], END, freq=ME)).ffill()
    return rf[rf.index >= START].dropna().rename("RF_CHF")


def to_chf(r, ccy, fx_ret):
    if ccy == "CHF":
        return r
    return ((1 + r) * (1 + fx_ret[ccy].reindex(r.index)) - 1).dropna()


def build():
    fx_ret = fx_chf().pct_change(fill_method=None)
    (IN / "msci").mkdir(exist_ok=True)
    check_msci_files()

    assets = {}
    for name, (src, sid, ccy) in ASSETS.items():
        if src == "msci":
            px = msci(sid)
            if px is None:
                print(f"!! {name}: no Gross/USD MSCI file with a '{sid}' column in {IN / 'msci'}")
                continue
            r = monthly(px)
        elif src == "investing":
            if not (IN / "investing" / sid).exists():
                print(f"!! {name}: file {sid} missing in {IN / 'investing'}")
                continue
            r = monthly(investing(sid))
        else:
            px = yahoo_tr(sid)
            if px is None:
                print(f"!! {name}: no Yahoo data for {sid}")
                continue
            r = monthly(px)
        assets[name] = to_chf(r, ccy, fx_ret)
        print(f"OK {name:<20} {sid:<16} from {assets[name].index[0]:%Y-%m}")
    assets = pd.DataFrame(assets)

    stocks = {}
    for name, tic in STOCKS.items():
        px = yahoo_tr(tic)
        if px is None:
            print(f"!! {name}: no Yahoo data for {tic}")
            continue
        r = monthly(px)
        if name in STOCK_FIRST_MONTH:
            r = r[r.index >= STOCK_FIRST_MONTH[name]]
        for m in STOCK_BAD_MONTHS.get(name, []):
            r[m] = np.nan
        stocks[name] = r
        print(f"OK {name:<20} {tic:<16} from {stocks[name].index[0]:%Y-%m}")
    stocks = pd.DataFrame(stocks)

    rf = rf_chf()
    assets.to_csv(OUT / "assets_returns_chf_monthly.csv")
    stocks.to_csv(OUT / "smi_stocks_returns_chf_monthly.csv")
    rf.to_csv(OUT / "rf_chf_monthly.csv")

    # when each asset has HISTORY_MONTHS of returns and can enter the optimisation
    avail = pd.DataFrame({
        "first_return": assets.apply(lambda s: s.first_valid_index()),
        "months": assets.notna().sum(),
        "tradable_from": assets.apply(lambda s: s.dropna().index[HISTORY_MONTHS]
                                      if s.notna().sum() > HISTORY_MONTHS else pd.NaT),
    }).sort_values("first_return")
    avail.to_csv(OUT / "asset_availability.csv")
    print(f"\nAsset availability ({HISTORY_MONTHS} months needed before trading):")
    print(avail.assign(first_return=avail.first_return.dt.strftime("%Y-%m"),
                       tradable_from=avail.tradable_from.dt.strftime("%Y-%m")).to_string())
    print(f"\nSaved to {OUT}")
    print(f"Risk-free: {rf.index[0]:%Y-%m} -> {rf.index[-1]:%Y-%m}")


if __name__ == "__main__":
    build()
