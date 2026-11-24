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


# removing artifacts left by pandas incompatible by polars
def pandas_csv_to_polars_cleaner(path=UNIVERSE_DATA_PATH / "smi_stocks_returns_chf_monthly.csv",
                                 rf_path=UNIVERSE_DATA_PATH / "rf_CHF_monthly.csv", other_asset_path = UNIVERSE_DATA_PATH / "assets_returns_chf_monthly.csv"):
    # removing the artifacts of pandas
    df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=None)
    df = df.rename({df.columns[0]: "Date"})
    rf_df = pl.read_csv(rf_path, try_parse_dates=True, infer_schema_length=None)
    rf_df = rf_df.rename({rf_df.columns[0]: "Date"})
    other_asset_df = pl.read_csv(other_asset_path, try_parse_dates=True, infer_schema_length=None)
    other_asset_df = other_asset_df.rename({other_asset_df.columns[0]: "Date"})

    # joining in one clean polars dataframe.
    df = df.join(rf_df, on="Date", how="left")
    df = df.join(other_asset_df, on="Date", how="left")

    # saving into Master as Master in parquet format
    df.write_parquet(MASTER_DATA_PATH / "Master.parquet")
    print(df.columns)
    del df, rf_df

#Running the code
if __name__ == "__main__":
    monthly_data(extension = ".csv", path = DAILY_DATA_PATH)
    pandas_csv_to_polars_cleaner()