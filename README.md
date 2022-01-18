# ml-pipeline

This repository shows how to use MLflow tracking, projects, and models modules. 
# Structure of this project

- [`logs/train_model.log`](https://github.com/haythemtellili/ML_pipeline/blob/master/logs/train_model.log "train_model.log") : file to log errors
- [`MLproject`](https://github.com/haythemtellili/ML_pipeline/blob/master/MLproject "MLproject") specifies the conda environment to run the project and defines `command` and `parameters` in `entry_points`
- [`Dockerfile`](https://github.com/haythemtellili/ML_pipeline/blob/master/Dockerfile "Dockerfile") used to build the image to serve the model 
- [`conda.yaml`](https://github.com/haythemtellili/ML_pipeline/blob/master/conda.yaml "conda.yaml") used to create virtual environment referenced by the `MLproject` 
- [`mlflow_model_driver.py`](https://github.com/haythemtellili/ML_pipeline/blob/master/mlflow_model_driver.py "mlflow_model_driver.py"): finds best training run and starts a REST API model server based on [MLflow Models](https://www.mlflow.org/docs/latest/models.html) in docker containers. 
- [`train.py`](https://github.com/haythemtellili/ML_pipeline/blob/master/train.py "train.py") : contains a file that trains a scikit-learn model and uses MLflow Tracking APIs to log the model and its metadata (e.g., hyperparameters and metrics)
- [`preprocess.py`](https://github.com/haythemtellili/ML_pipeline/blob/master/preprocess.py "preprocess.py") : contains a file that perform data processing
- [`utils.py`](https://github.com/haythemtellili/ML_pipeline/blob/master/utils.py "utils.py") : contains a file with helper functions
- [`main.py`](https://github.com/haythemtellili/ML_pipeline/blob/master/main.py "main.py") : contains a file where we orchestrate everything into one worflow
- [`data/raw/hour.csv`](https://github.com/haythemtellili/ML_pipeline/blob/master/data/raw/hour.csv "hour.csv"): contains raw data  [Bike Sharing Dataset Data Set](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) 
- [`data/processed/data_preprocessed.csv`](https://github.com/haythemtellili/ML_pipeline/blob/master/data/processed/data_preprocessed.csv "data_preprocessed.csv"): contains processed data
# Getting started
Prerequisites: 

 1. Python 3
 2. Install Docker per instructions at https://docs.docker.com/install/overview/
 3. Install Anaconda 

## Model training with MLflow:

 1. clone this repo: `git clone https://github.com/haythemtellili/ML_pipeline.git`
 2. Enter to ML_pipeline: `cd ML_pipeline`
 3. build the image for the project's Docker container environment: `docker build -t mlflow_example -f Dockerfile .`
 4. create conda environment: `conda env create --name ml-pipeline --file=conda.yaml`
 5. activate conda environment: `conda activate ml-pipeline`
 6. Run the workflow: `mlflow run .`

## Run MLflow tracking UI:
In the same repo directory, run `mlflow ui --host 0.0.0.0 --port 5000`
UI is accessible at http://localhost:5000/


## Dockerized MLflow model serving (REST API)
In the same repo directory, run `python3 mlflow_model_driver.py`

## Inference request:
```bash
curl --silent --show-error 'http://localhost:5001/invocations' -H 'Content-Type: application/json' -d '{
    "columns": ["year","hour_of_day","is_holiday","weekday","is_workingday","temperature","feels_like_temperature","humidity","windspeed","year.1","dayofweek","year_season","hour_workingday_casual","hour_workingday_registered","count_season","season_1","season_2","season_3","season_4","weathersit_1","weathersit_2","weathersit_3","weathersit_4","mnth_1","mnth_2","mnth_3","mnth_4","mnth_5","mnth_6","mnth_7","mnth_8","mnth_9","mnth_10","mnth_11","mnth_12"
],
    "data": [[0,-1.670003982455765,0,6,0,-1.3346475857785418,-1.0932806043146361,0.9473724999661597,-1.5538885118643786,2011,5,2011.1,0,0,56.0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]]
}'
```
