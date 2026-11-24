import os
from typing import Final
from pathlib import Path
import polars as pl

#Config and paths
PROJECT_PATH: Final[Path] = Path(__file__).parent.parent
DATA_PATH: Final[Path] = PROJECT_PATH / "Data_Output"
DAILY_DATA_PATH: Final[Path] = DATA_PATH / "daily"
MONTHLY_DATA_PATH: Final[Path] = DATA_PATH / "Monthly"
MASTER_DATA_PATH: Final[Path] = DATA_PATH / "Master"

#Tickers used
TICKER_LIST:Final = ["^RUT","AAPL","MSFT"]
#Dates
START_DATE: Final = pl.date(year=1960, month=1, day=1)
END_DATE: Final = pl.date(year=2025, month=1, day=1)

#Steps, Obs, ETC
STEPS:Final = 1
MIN_OBS:Final = 24
MAX_OBS:Final[bool] = False #False means no maximum and is expending window

def create_folders():
    os.makedirs(DAILY_DATA_PATH, exist_ok=True)
    os.makedirs(MONTHLY_DATA_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(MASTER_DATA_PATH, exist_ok=True)

create_folders()

