from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import structlog
import typer


logger = structlog.get_logger()


app = typer.Typer()


def corpus_from_path(path: Path) -> tuple[str, ...]:
    corpus: list[str] = []
    for f in path.iterdir():
        if f.is_file():
            with open(f) as fh:
                corpus.append(fh.read())
    return tuple(corpus)


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


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

    logger.info("using stop_words", stop_words=stop_words or None)
    
    model = TfidfVectorizer(stop_words=stop_words or None)
    X = model.fit_transform(corpus).todense()
    query_vec = model.transform([query]).todense()
    sim = cosine_sim(query_vec, X.T)
    highest_sim = sim.argmax()
    print(corpus[highest_sim])


if __name__ == "__main__":
    app()
