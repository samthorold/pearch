from pathlib import Path
import string
import sys
from typing import Callable, Optional

import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import structlog
import typer


logger = structlog.get_logger()


STOPWORDS = nltk.corpus.stopwords.words("english")
STEMMER = SnowballStemmer("english")


app = typer.Typer()


def corpus_from_path(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    corpus: list[str] = []
    files: list[Path] = []
    for f in path.iterdir():
        if f.is_file():
            files.append(f)
            with open(f) as fh:
                corpus.append(fh.read())
    return tuple(corpus), tuple(files)


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


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


@app.command()
def stem(path: Path, out: Optional[Path] = None) -> None:
    logger.info("stemming", path=path)
    if not path.exists():
        logger.error("non-existant path", path=path)
        sys.exit(1)

    stemmer = SnowballStemmer("english")
    logger.info("using stemmer", stemmer=stemmer)

    if path.is_file():
        paths = [path]
    else:
        paths = list(path.iterdir())

    if out is not None and not out.exists():
        logger.info("creating out dir", out=out)
        out.mkdir()

    for f in paths:
        if not f.is_file():
            continue
        logger.info("stemming", file=f)
        with open(f) as fh:
            doc = fh.read()
        doc = _stem_document(doc=doc, stem=stemmer.stem)
        logger.info("stemmed", file=f)
        if out is None:
            print(doc)
        else:
            with open(out / f.name, "w") as fh:
                fh.write(doc)


@app.command()
def search(query: str, path: Path = Path("stemmed-docs")) -> None:
    logger.info("searching", query=query, path=path)
    query = _stem_document(doc=query, stem=STEMMER.stem)
    if not path.exists():
        logger.error("non-existant path", path=path)
        sys.exit(1)
    if not path.is_dir():
        logger.error("path not a dir", path=path)
        sys.exit(1)

    corpus, files = corpus_from_path(path)
    logger.info("loaded corpus", docs=len(corpus))
    
    model = TfidfVectorizer()
    X = model.fit_transform(corpus).todense()
    logger.info("fitted model", model=model)
    query_vec = model.transform([query]).todense()
    sim = np.array(cosine_sim(query_vec, X.T)).flatten()
    sim = sim.argsort()[::-1][:5]
    logger.info("closest match", score=sim)
    files = [files[i].name for i in sim]
    print("\n".join(files))


if __name__ == "__main__":
    app()
