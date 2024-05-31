import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import numpy as np


def load_model_data():
    """Load the model data from the parquet file
    target: toxicity or class
    features: comment_vec or comment_text_prep
    """
    file_sampled = r"src\txt_comments\jigsaw_data_prep_sampled.parquet"
    return pd.read_parquet(file_sampled)


def get_train_test_split(test_size: float = 0.2, seed: int = 42) -> tuple:
    """Get the train test split"""
    df = load_model_data()
    features = df["comment_vec"].to_list()
    target = df["class"].to_list()
    return train_test_split(features, target, test_size=test_size, random_state=seed)
