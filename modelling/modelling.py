import sys
import os

import dagshub
import mlflow

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocessing.automate_evanhanif import preprocess_data

dagshub.init(repo_owner="evanhfw", repo_name="SMSL", mlflow=True)
mlflow.autolog()

X_train, X_test, y_train, y_test, PreprocessingPipeline = preprocess_data(
    "../data/raw/Credit Score Dataset.csv",
    "../data/preprocessed/pipeline.pkl",
    "Credit_Score",
    "../data/preprocessed/headers.csv",
)
