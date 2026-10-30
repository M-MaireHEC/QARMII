import os
from typing import Final
from pathlib import Path

#Config and paths
PROJECT_PATH: Final[Path] = Path(__file__).parent.parent
DATA_PATH: Final[Path] = PROJECT_PATH / "Data_Output"
DAILY_DATA_PATH: Final[Path] = DATA_PATH / "daily"
MONTHLY_DATA_PATH: Final[Path] = DATA_PATH / "Monthly"
MASTER_DATA_PATH: Final[Path] = DATA_PATH / "Master"

def create_folders():
    os.makedirs(DAILY_DATA_PATH, exist_ok=True)
    os.makedirs(MONTHLY_DATA_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(MASTER_DATA_PATH, exist_ok=True)

create_folders()

