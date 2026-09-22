"""
Pluggable sentence-embedding backends.

Two options, same interface (fit on training texts if needed, then encode
any list of strings into a fixed-size float array):

  "sentence-transformer" (recommended, what model.py was designed around):
      a real pretrained embedding model that understands English generally
      -- it should recognize a paraphrase it never saw in training (e.g.
      "surrounded by mountains" and "encircled by peaks" land close
      together in embedding space even though only one appeared in the
      synthetic dataset). Needs `sentence-transformers` installed and a
      one-time internet download of the model weights (~80MB, cached
      locally afterward -- offline from then on).

  "tfidf" (offline fallback, no downloads, no extra install beyond
      scikit-learn which you already have):
      classical bag-of-words weighted by term frequency. No general
      language understanding -- it only "knows" the exact words it saw
      during fitting, so it won't generalize to a synonym it's never
      seen. Given how templated the synthetic dataset's vocabulary is,
      this actually performs surprisingly well on THIS dataset, but it
      will be noticeably worse than the sentence-transformer on real,
      free-form phrasing at inference time (a genuinely new word or
      phrasing it never saw during fitting gets no signal at all).

Use sentence-transformer for the version you actually ship/demo. Use
tfidf if you want a zero-dependency, fully-offline version, or while
developing without internet access.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class TfidfEmbedder:
    kind = "tfidf"

    def __init__(self, max_features: int = 400):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self._fitted = False

    @property
    def embedding_dim(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode()")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: str) -> "TfidfEmbedder":
        obj = cls.__new__(cls)
        obj.kind = "tfidf"
        with open(path, "rb") as f:
            obj.vectorizer = pickle.load(f)
        obj._fitted = True
        return obj


class SentenceTransformerEmbedder:
    kind = "sentence-transformer"
    embedding_dim = 384  # fixed for all-MiniLM-L6-v2

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def fit(self, texts: list[str]):
        return self  # nothing to fit -- it's pretrained and frozen

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def save(self, path: str):
        # nothing to save -- reloadable by model_name alone
        with open(path, "w") as f:
            f.write(self.model_name)

    @classmethod
    def load(cls, path: str) -> "SentenceTransformerEmbedder":
        with open(path) as f:
            model_name = f.read().strip()
        return cls(model_name)


def get_embedder(kind: str, **kwargs):
    if kind == "tfidf":
        return TfidfEmbedder(**kwargs)
    if kind == "sentence-transformer":
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedder kind {kind!r}, choose 'tfidf' or 'sentence-transformer'")


def load_embedder(kind: str, path: str):
    if kind == "tfidf":
        return TfidfEmbedder.load(path)
    if kind == "sentence-transformer":
        return SentenceTransformerEmbedder.load(path)
    raise ValueError(f"unknown embedder kind {kind!r}")
