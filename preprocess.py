import mlflow
import os
import pandas as pd
from sklearn import preprocessing

from utils import dummify_dataset

if __name__ == "__main__":

    with mlflow.start_run(run_name="load_raw_data") as run:

        mlflow.set_tag("mlflow.runName", "load_raw_data")
        # Read the data csv file
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/raw/hour.csv"
        )
        # load input data into pandas dataframe
        bikes = pd.read_csv(data_path)

        ##############################
        #     Feature Engineering
        ##############################
        date = pd.DatetimeIndex(bikes["dteday"])

        bikes["year"] = date.year

        bikes["dayofweek"] = date.dayofweek

        bikes["year_season"] = bikes["year"] + bikes["season"] / 10

        bikes["hour_workingday_casual"] = bikes[["hr", "workingday"]].apply(
            lambda x: int(10 <= x["hr"] <= 19), axis=1
        )

        bikes["hour_workingday_registered"] = bikes[["hr",
                                                     "workingday"]].apply(
            lambda x: int(
                (
                 x["workingday"] == 1 and (x["hr"] == 8 or 17 <= x["hr"] <= 18)
                )
                or (x["workingday"] == 0 and 10 <= x["hr"] <= 19)
            ),
            axis=1,
        )

        by_season = bikes.groupby("year_season")[["cnt"]].median()
        by_season.columns = ["count_season"]

        bikes = bikes.join(by_season, on="year_season")

        # One-Hot-Encoding
        columns_to_dummify = ["season", "weathersit", "mnth"]
        for column in columns_to_dummify:
            bikes = dummify_dataset(bikes, column)

        # Normalize features - scale
        numerical_features = ["temp", "atemp", "hum", "windspeed", "hr"]
        bikes.loc[:, numerical_features] = preprocessing.scale(
            bikes.loc[:, numerical_features]
        )

        # remove unused columns
        bikes.drop(columns=["instant", "dteday", "registered", "casual"],
                   inplace=True)

        # use better column names
        bikes.rename(
            columns={
                "yr": "year",
                "mnth": "month",
                "hr": "hour_of_day",
                "holiday": "is_holiday",
                "workingday": "is_workingday",
                "weathersit": "weather_situation",
                "temp": "temperature",
                "atemp": "feels_like_temperature",
                "hum": "humidity",
                "cnt": "rented_bikes",
            },
            inplace=True,
        )

        bikes.to_csv("./data/processed/data_preprocessed.csv", index=False)
