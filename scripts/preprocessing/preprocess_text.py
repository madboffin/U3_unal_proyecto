""" funciones de procesamiento de texto"""

import re

from typing import Iterable
from unidecode import unidecode
import spacy


def to_spacy(corpus: Iterable, nlp: spacy.language.Language):
    """Convert text to spacy doc"""
    return list(nlp.pipe(corpus, n_process=1))


def filter_stopwords_and_len(doc: spacy.tokens.Doc, len_min):
    """Remove stopwords and tokens with length less than len_min"""
    return filter(lambda token: not token.is_stop and len(token) >= len_min, doc)


def lemmatize(doc: spacy.tokens.Doc):
    """Lemmatize text"""
    return filter(lambda token: token.lemma_, doc)
    # return [token.lemma_ for token in doc]


def get_text_from_doc(doc: spacy.tokens.Doc):
    """Get text from spacy doc"""
    return " ".join(token.text for token in doc)


def normalize(text: str):
    """Convert to lowercase and remove accents"""
    return unidecode(text).lower()


def remove_nonalpha(text: str) -> str:
    """Remove non-alphabetic characters"""
    return re.sub(r"[^a-z ]", " ", text)


def remove_doublespaces(text: str):
    """Remove extra whitespaces"""
    return " ".join(text.split())


def preprocess_text(doc, lemma: bool = False) -> Iterable:
    """Preprocess text"""
    doc_filter = filter_stopwords_and_len(doc, 3)
    doc_filter = lemmatize(doc_filter) if lemma else doc_filter
    text = get_text_from_doc(doc_filter)
    text = remove_nonalpha(text)
    text = remove_doublespaces(text)
    return text
