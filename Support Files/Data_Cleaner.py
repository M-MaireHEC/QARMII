import polars as pl
import pathlib
from Config import *


#Getting the names of files inside the folder
def get_list_name(extension:str = ".csv", path:pathlib.Path = DAILY_DATA_PATH):
    names = []
    for name in list(path.glob('*' + extension)):
        names.append(name.name)
    return names

#Changing the values into monthly observations
def monthly_data(extension:str = ".csv", path:pathlib.Path = DAILY_DATA_PATH):
    names_list = get_list_name(extension, path)
    for name in names_list:
        df = pl.read_csv(path/name)

        df = df.with_columns(
            pl.col("Date").str.slice(0, 10).str.to_date("%Y-%m-%d")
        )

        df = df.sort("Date").filter(
            pl.col("Date") == pl.col("Date").max().over(pl.col("Date").dt.year(), pl.col("Date").dt.month())
        )

        file_name = name.replace(extension, ".parquet")
        df.write_parquet(MONTHLY_DATA_PATH/file_name)

#Running the code
if __name__ == "__main__":
    monthly_data(extension = ".csv", path = DAILY_DATA_PATH)