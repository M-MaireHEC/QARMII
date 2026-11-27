import os
import datetime as dt
from typing import Final
from pathlib import Path
import polars as pl

#Config and paths
PROJECT_PATH: Final[Path] = Path(__file__).parent.parent
DATA_PATH: Final[Path] = PROJECT_PATH / "Data_Output"
DAILY_DATA_PATH: Final[Path] = DATA_PATH / "daily"
MONTHLY_DATA_PATH: Final[Path] = DATA_PATH / "Monthly"
MASTER_DATA_PATH: Final[Path] = DATA_PATH / "Master"
UNIVERSE_DATA_PATH: Final[Path] = DATA_PATH / "universe"

#Dates
START_DATE: Final = pl.date(year=1960, month=1, day=1)
END_DATE: Final = pl.date(year=2025, month=1, day=1)

#Tickers used
ASSETS:Final = {
    "Gold":          ("investing", "xau_usd.csv", "USD"),           # spot, 1990
    "Silver":        ("investing", "xag_usd.csv", "USD"),           # spot, 1990
    "US Gov 10y":    ("investing", "tr_us10.csv", "USD"),           # TR US 10Y Gov Benchmark, 1990
    #"US Gov 30y":    ("investing", "tr_us30.csv", "USD"),           # TR US 30Y Gov Benchmark, 1990
    "CH Gov 2y":     ("investing", "tr_ch2.csv", "CHF"),            # TR SW 2Y Gov Benchmark, 1990
    #"CH Gov 10y":    ("investing", "tr_ch10.csv", "CHF"),           # TR SW 10Y Gov Benchmark, 1990
    "DE Gov 10y":    ("investing", "tr_de10.csv", "EUR"),           # TR Germany 10Y Gov Benchmark, 1990
    "UK Gov 10y":    ("investing", "tr_uk10.csv", "GBP"),           # TR UK 10Y Gov Benchmark, 1990
    "US Corp IG":    ("yahoo", "VWESX", "USD"),     # Vanguard Long-Term Investment-Grade fund, 1990
    "US Corp HY":    ("yahoo", "VWEHX", "USD"),     # Vanguard High-Yield Corporate fund, 1990
    "US Equity":     ("yahoo", "^SP500TR", "USD"),                  # S&P 500 Total Return, 1990
    "CH Equity":     ("investing", "spi_tr.csv", "CHF"),            # SPI (Swiss Performance Index, TR), 1990
    "DE Equity":     ("yahoo", "^GDAXI", "EUR"),                    # DAX (total return), 1990
    "FR Equity":     ("investing", "cac40_tr.csv", "EUR"),          # CAC 40 Gross Total Return, 1990
    "UK Equity":     ("investing", "ftse_allshare_tr.csv", "GBP"),  # FTSE All-Share TR, 1994
    "US Small Cap":  ("msci", "USA Small Cap", "USD"),              # MSCI USA Small Cap Gross, 2001
}

# SMI after the Sep 2026 review (Amrize left out: too short) + former members still listed.
# Credit Suisse is not on Yahoo any more (survivorship bias: mention it in the report).
STOCKS:Final = {
    "ABB": "ABBN.SW", "Alcon": "ALC.SW", "Galderma": "GALD.SW", "Geberit": "GEBN.SW",
    "Givaudan": "GIVN.SW", "Holcim": "HOLN.SW", "Logitech": "LOGN.SW", "Lonza": "LONN.SW",
    "Nestle": "NESN.SW", "Novartis": "NOVN.SW", "Partners Group": "PGHN.SW", "Richemont": "CFR.SW",
    "Roche": "RO.SW", "Sandoz": "SDZ.SW", "Sika": "SIKA.SW", "Swiss Life": "SLHN.SW",
    "Swiss Re": "SREN.SW", "UBS": "UBSG.SW", "Zurich Insurance": "ZURN.SW",
    # former members
    "Swisscom": "SCMN.SW", "Kuehne+Nagel": "KNIN.SW", "Sonova": "SOON.SW", "SGS": "SGSN.SW",
    "Swatch": "UHR.SW", "Adecco": "ADEN.SW",
}

# Yahoo errors confirmed against investing.com (no estimates, the bad months are left empty):
STOCK_FIRST_MONTH:Final = {"Givaudan": "2000-07", "Geberit": "1999-07"}  # Yahoo has pre-listing prices
STOCK_BAD_MONTHS:Final = {"Logitech": ["2001-01"]}

TICKER_LIST:Final = list(ASSETS.keys())+list(STOCKS.keys())

#Market caps for Black-Litterman. Real file if it exists, otherwise a DUMMY:
#every ticker gets the same cap every month -> w_mkt = equal weights over the tradable tickers.
MARKET_CAPS_PATH: Final[Path] = MASTER_DATA_PATH / "market_caps.parquet"
DUMMY_MARKET_CAP: Final = 1_000_000_000

def dummy_market_caps() -> pl.DataFrame:
    dates = pl.date_range(dt.date(1960, 1, 1), dt.date.today(), interval="1mo", eager=True)
    return pl.DataFrame({"Date": dates}).with_columns(
        pl.lit(DUMMY_MARKET_CAP, dtype=pl.Float64).alias(t) for t in TICKER_LIST
    )

if MARKET_CAPS_PATH.is_file():
    MARKET_CAPS: Final = pl.read_parquet(MARKET_CAPS_PATH)
else:
    print(f"WARNING: {MARKET_CAPS_PATH.name} not found, using DUMMY equal market caps")
    MARKET_CAPS: Final = dummy_market_caps()
BL_MARKET_SHARPE:Final = 0.25






#Steps, Obs, ETC
STEPS:Final = 1
MIN_OBS:Final = 120 # months of data an asset needs before it can be traded (10 years)
MAX_OBS:Final[bool] = False #False means no maximum and is expending window

def create_folders():
    os.makedirs(DAILY_DATA_PATH, exist_ok=True)
    os.makedirs(MONTHLY_DATA_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(MASTER_DATA_PATH, exist_ok=True)

create_folders()

