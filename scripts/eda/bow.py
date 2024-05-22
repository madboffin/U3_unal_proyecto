from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_bow(corpus: Iterable):
    vect = CountVectorizer(max_features=1000).fit(corpus)
    X = vect.transform(corpus).toarray()
    return X, vect


def get_wordcloud(doc_array: np.ndarray, model: CountVectorizer):
    vocab = model.get_feature_names_out()
    count_dict = dict(zip(vocab, doc_array.sum(axis=0)))
    wc = WordCloud(width=800, height=400, background_color="white")
    wc.generate_from_frequencies(count_dict)
    return wc


def plot_wordcloud(doc_array, model) -> None:
    """Plot wordcloud from bag of words matrix"""
    wc = get_wordcloud(doc_array, model)
    _fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")


def plot_word_distribution(doc_array, model, title: str) -> None:
    """Plot word distribution from bag of words matrix"""
    vocab = model.get_feature_names_out()
    count_dict = dict(zip(vocab, doc_array.sum(axis=0)))
    count_df = pd.DataFrame.from_dict(count_dict, orient="index", columns=["count"])
    count_df = count_df.sort_values(by="count", ascending=False)
    count_df = count_df.head(20)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(count_df.index, count_df["count"])
    ax.set_title(title)
    ax.set_xlabel("Word")
    ax.set_ylabel("Count")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
