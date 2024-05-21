"""funciones de preprocesamiento de df de datos"""

from typing import Iterable

import pandas as pd


def tf_to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    return df


def drop_missing_comments(df: pd.DataFrame):
    return df[df["comment_text"].notna()]


def drop_short_comments(df: pd.DataFrame, min_len: int = 4):
    return df[df["comment_text"].str.len() > min_len]


def preprocess_df(df: pd.DataFrame, to_numeric_col: list[str]) -> pd.DataFrame:
    """convert columns to numeric, add length of text and convert created_date to datetime"""
    df = drop_missing_comments(df)
    df = drop_short_comments(df)
    df = tf_to_numeric(df, to_numeric_col)
    df["len_text"] = df["comment_text"].str.len()
    df["created_date"] = pd.to_datetime(df["created_date"].str.slice(0, 10))
    return df


def sample_from_trainset(df: pd.DataFrame, n: int = 5_000, seed: int = 42):
    """sample n rows from the train set"""
    return df.query("split=='train'").sample(n, random_state=seed)


def sample_with_quota(df: pd.DataFrame, n_by_quota: int = 5_000):
    """sample (n_by_quota * 5) rows in the train set, using 5 toxicity levelss"""
    quota_sampling = []
    n_samples = 5
    for k in range(n_samples):
        start = 1 / n_samples * k
        end = 1 / n_samples * (k + 1)
        quota_sampling.append(
            df.query(f"{start} <= toxicity <= {end}").sample(n_by_quota)
        )

    return pd.concat(quota_sampling)
