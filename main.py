from pathlib import Path
import string
import sys
from typing import Callable

import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import structlog
import typer


logger = structlog.get_logger()


STOPWORDS = nltk.corpus.stopwords.words("english")


app = typer.Typer()


def corpus_from_path(path: Path) -> tuple[str, ...]:
    corpus: list[str] = []
    for f in path.iterdir():
        if f.is_file():
            with open(f) as fh:
                corpus.append(fh.read())
    return tuple(corpus)


def clean_word(word: str) -> str:
    return "".join([c for c in word.lower() if c in string.ascii_lowercase])


def _stem_document(doc: str, stem: Callable[[str], str]) -> str:
    s = ""
    for w in nltk.word_tokenize(doc):
        word = clean_word(w)
        if word and word not in STOPWORDS:
            stemmed = stem(word).strip()
            s += f" {stemmed}"
    return s.strip()


def stem_corpus(corpus: tuple[str, ...]) -> tuple[str, ...]:
    stemmer = SnowballStemmer("english")
    corp: list[str] = []
    for c in corpus:
        corp.append(stem_document(doc=c, stem=stemmer.stem))
        breakpoint()
    return tuple(corp)


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


@app.command()
def stem(path: Path, print_original: bool = False) -> None:
    logger.info("stemming", path=path)
    if not path.exists():
        logger.error("non-existant path", path=path)
        sys.exit(1)
    if not path.is_file():
        logger.error("path not a file", path=path)
        sys.exit(1)

    with open(path) as fh:
        doc = fh.read()

    if print_original:
        print(doc)

    stemmer = SnowballStemmer("english")

    print(_stem_document(doc=doc, stem=stemmer.stem))


@app.command()
def index(query: str, path: Path, stop_words: str = "") -> None:
    logger.info("indexing", path=path)
    if not path.exists():
        logger.error("non-existant path", path=path)
        sys.exit(1)
    if not path.is_dir():
        logger.error("path not a dir", path=path)
        sys.exit(1)

    corpus = corpus_from_path(path)
    corpus = stem_corpus(corpus)

    logger.info("using stop_words", stop_words=stop_words or None)
    
    model = TfidfVectorizer(stop_words=stop_words or None)
    X = model.fit_transform(corpus).todense()
    query_vec = model.transform([query]).todense()
    sim = cosine_sim(query_vec, X.T)
    highest_sim = sim.argmax()
    print(corpus[highest_sim])


if __name__ == "__main__":
    app()
