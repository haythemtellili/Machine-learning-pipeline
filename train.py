import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from utils import rmse_score, rmse_cv_score

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")
np.random.seed(42)

# create model_artifacts directory
model_artifacts_dir = "/tmp/model_artifacts"
Path(model_artifacts_dir).mkdir(exist_ok=True)

# Read the data csv file
data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/processed/data_preprocessed.csv"
)
# load input data into pandas dataframe
bike_sharing = pd.read_csv(data_path)
# Split the dataset randomly into 70% for training and 30% for testing.
X = bike_sharing.drop("rented_bikes", axis=1)
y = bike_sharing.rented_bikes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=42
)

# main entry point
if __name__ == "__main__":
    # parse run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    run_parameters = vars(parser.parse_args())
    with mlflow.start_run(run_name="train") as run:

        mlflow.set_tag("mlflow.runName", "train")

        # create model instance: GBRT (Gradient Boosted Regression Tree)
        model = GradientBoostingRegressor(**run_parameters)

        # Model Training
        model.fit(X_train, y_train)

        # get evaluations scores
        score = rmse_score(y_test, model.predict(X_test))
        score_cv = rmse_cv_score(model, X_train, y_train)

        # generate charts
        # model_feature_importance(model, X_train, model_artifacts_dir)

        # log input features
        mlflow.set_tag("features", str(X_train.columns.values.tolist()))

        # Log tracked parameters
        mlflow.log_params(run_parameters)

        mlflow.log_metrics(
            {
                "RMSE_CV": score_cv.mean(),
                "RMSE": score,
            }
        )

        # log training loss
        for s in model.train_score_:
            mlflow.log_metric("Train Loss", s)

        # get model signature
        signature = infer_signature(
            model_input=X_train,
            model_output=model.predict(X_train)
        )

        # Save model to artifacts
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # log charts
        # mlflow.log_artifacts(model_artifacts_dir)
        # Write metrics to file
        with open('metrics.txt', 'w') as outfile:
            outfile.write(f'\nRoot Mean Square Error = {score}.')
