# QARM II data set

Monthly total returns in **CHF** for a Swiss long-term investor, February 1990 to the last complete month.
Built by `Support Files/fetch_universe.py` from the raw files in `Data_Input/`.

## Files (Data_Output/universe)

| File | Content |
|---|---|
| `assets_returns_chf_monthly.csv` | 16 asset classes, monthly simple returns in CHF (month-end dates) |
| `rf_chf_monthly.csv` | CHF risk-free return per month (column `RF_CHF`) |
| `asset_availability.csv` | first return of each asset and `tradable_from` = first month with 120 months of history |
| `smi_stocks_returns_chf_monthly.csv` | 25 Swiss stocks (current and former SMI members), monthly total returns in CHF |

Empty cells mean "no data yet" (the asset did not exist or is not covered). Do not fill them with 0.

```python
import pandas as pd
D = "Data_Output/universe/"
R = pd.read_csv(D + "assets_returns_chf_monthly.csv", index_col=0, parse_dates=True)
rf = pd.read_csv(D + "rf_chf_monthly.csv", index_col=0, parse_dates=True)["RF_CHF"]
avail = pd.read_csv(D + "asset_availability.csv", index_col=0, parse_dates=["first_return", "tradable_from"])
excess = R.sub(rf, axis=0)                                  # excess returns over the CHF risk-free rate
universe_at = lambda t: avail.index[avail.tradable_from <= t].tolist()   # assets allowed at date t
```

## Assets and sources (one source per asset, all total return)

| Asset | Source | Data from |
|---|---|---|
| Gold, Silver | investing.com spot XAU/USD, XAG/USD | 1990 |
| US Gov 10y, US Gov 30y | investing.com, Thomson Reuters US Government Benchmark TR indices | 1990 |
| CH Gov 2y, CH Gov 10y | investing.com, Thomson Reuters Switzerland Government Benchmark TR indices | 1990 |
| DE Gov 10y | investing.com, Thomson Reuters Germany 10Y Government Benchmark TR | 1990 |
| UK Gov 10y | investing.com, Thomson Reuters UK 10Y Government Benchmark TR | 1990 |
| US Corp IG | Yahoo VWESX (Vanguard Long-Term Investment-Grade fund) | 1990 |
| US Corp HY | Yahoo VWEHX (Vanguard High-Yield Corporate fund) | 1990 |
| US Equity | Yahoo ^SP500TR (S&P 500 Total Return) | 1990 |
| CH Equity | investing.com SPI (Swiss Performance Index, total return) | 1990-05 |
| DE Equity | Yahoo ^GDAXI (DAX, a total-return index) | 1990 |
| FR Equity | investing.com CAC 40 Gross Total Return | 1990 |
| UK Equity | investing.com FTSE All-Share Total Return | 1994 |
| US Small Cap | MSCI USA Small Cap, Gross, USD (msci.com) | 2001 |
| Swiss stocks | Yahoo prices + Yahoo dividend records | 1990 to 2024 depending on listing |

Currency conversion only (FX is not an asset): FRED DEXSZUS, DEXUSUK, DEXUSEU; before 1999 EUR via the
Deutsche Mark (EXSZUS / EXGEUS x 1.95583). Risk-free: FRED 3-month CHF interbank rate (IR3TIB01CHM156N),
before mid-1999 the 3-month CHF euro-deposit rate (IR3TED01CHM156N); the rate of month t-1 is earned in month t.

## Processing (everything the script does to the data)

1. Month-end value of each series, simple return, then converted to CHF: (1 + r_local)(1 + r_FX) - 1.
2. Yahoo series: total return = (close + dividend) / previous close. Dividends above 25% of the price are
   treated as data errors and ignored. London listings: dividends quoted in GBP are converted to pence.
3. Swiss stocks, Yahoo errors confirmed against investing.com: Givaudan starts 2000-07 and Geberit 1999-07
   (Yahoo has prices before their listing); Logitech 2001-01 is left empty (split not adjusted on Yahoo).
4. Nothing is estimated, interpolated or spliced.

## Known limitations (mention in the report)

- Government bonds: each index follows a single benchmark bond of that maturity, not a broad index.
- Credit: VWESX and VWEHX are actively managed funds, returns are after fees (about 0.2% per year).
- Swiss stocks: only companies still listed today (Credit Suisse, Syngenta, Actelion etc. are missing),
  so the stock list has survivorship bias. The SPI (CH Equity) does not.
- CH Equity 1991-10 has a 0% return (repeated value in the investing.com history).

## Refreshing the data

`pip install yfinance pandas numpy xlrd`, then run `Support Files/fetch_universe.py`. The investing.com and
MSCI files in `Data_Input/` were downloaded by hand and are not updated by the script.
