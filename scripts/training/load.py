import pandas as pd


def load_model_data():
    """Load the model data from the parquet file
    target: toxicity or class
    features: comment_vec or comment_text_prep
    """
    file_sampled = r"src\txt_comments\jigsaw_data_prep_sampled.parquet"
    return pd.read_parquet(file_sampled)
