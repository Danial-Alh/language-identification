import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import random


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def save_pickle(obj, filepath):
    folderpath = "/".join(filepath.split("/")[:-1])
    if folderpath:
        os.makedirs(folderpath, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


class ClassEncoder:
    def __init__(self, values=None, itos=None) -> None:
        assert not (values is None and itos is None)
        self.itos = list(sorted(set(values))) if itos is None else list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        path = path.rstrip("/")
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/itos.json", "w") as f:
            json.dump(self.itos, f)

    @staticmethod
    def load(path) -> "ClassEncoder":
        path = path.rstrip("/")
        with open(f"{path}/itos.json") as f:
            itos = json.load(f)
        return ClassEncoder(itos=itos)


class NaivePredictor:
    def __init__(
        self,
        tokenize_fn: Callable[[str], list[str]],
        lang_encoder: ClassEncoder,
        token_encoder: ClassEncoder | None = None,
        token_lang_scores: csr_matrix | None = None,
    ):
        self.tokenize_fn = tokenize_fn
        self.lang_encoder = lang_encoder
        self.token_encoder = token_encoder
        self.token_lang_scores = token_lang_scores

    def build_token2langs(self, df: pd.DataFrame):
        print("build token to langs mapping")
        token2langs: dict[str, list] = defaultdict(list)
        for rec in df.itertuples():
            for x in self.tokenize_fn(rec.sentence):
                token2langs[x].append(rec.label_str)
        return token2langs

    def build_token_lang_similarity_matrix(self, token2langs: dict[str, list]):
        print("build token-langs similarity matrix")
        token_lang_scores = []
        for token, langs in token2langs.items():
            for lang, score in Counter(langs).items():
                score /= len(langs)
                token_lang_scores.append(
                    (
                        score,
                        self.token_encoder.stoi[token],
                        self.lang_encoder.stoi[lang],
                    )
                )

        d, r, c = zip(*token_lang_scores)
        token_lang_scores = csr_matrix(
            (d, (r, c)),
            shape=(len(token2langs), len(self.lang_encoder)),
        )

        return token_lang_scores

    def fit(self, df: pd.DataFrame):
        token2langs = self.build_token2langs(df)
        self.token_encoder = ClassEncoder(token2langs.keys())
        self.token_lang_scores = self.build_token_lang_similarity_matrix(token2langs)

    def predict(self, sentences: list[str]) -> list[str]:
        sentences_vec = []
        for i, sentence in enumerate(sentences):
            for c, in_sentence_freq in Counter(
                [c for c in self.tokenize_fn(sentence) if c.strip()]
            ).items():
                if c not in self.token_encoder.stoi:
                    continue
                sentences_vec.append((in_sentence_freq, i, self.token_encoder.stoi[c]))
        d, r, c = zip(*sentences_vec)
        sentences_vec = csr_matrix(
            (d, (r, c)), shape=(len(sentences), len(self.token_encoder))
        )
        probs = (sentences_vec @ self.token_lang_scores).toarray()
        argmax_indices = np.argmax(probs, axis=-1)
        return [self.lang_encoder.itos[idx] for idx in argmax_indices]

    def save(self, path: str):
        path = path.rstrip("/")
        os.makedirs(path, exist_ok=True)
        save_pickle(self.tokenize_fn, f"{path}/tokenize-fn.pkl")
        self.lang_encoder.save(f"{path}/lang-encoder")
        self.token_encoder.save(f"{path}/token-encoder")
        np.save(f"{path}/token-lang-scores.npy", self.token_lang_scores)

    @staticmethod
    def load(path: str) -> "NaivePredictor":
        path = path.rstrip("/")
        tokenize_fn = load_pickle(f"{path}/tokenize-fn.pkl")
        lang_encoder = ClassEncoder.load(f"{path}/lang-encoder")
        token_encoder = ClassEncoder.load(f"{path}/token-encoder")
        token_lang_scores = np.load(f"{path}/token-lang-scores.npy")

        return NaivePredictor(
            tokenize_fn=tokenize_fn,
            lang_encoder=lang_encoder,
            token_encoder=token_encoder,
            token_lang_scores=token_lang_scores,
        )


def char_tokenizer(x: str):
    return x


def unigram_tokenizer(x: str):
    return x.split()


def subword_tokenizer(x: str, w: int):
    return [x[i : i + w] for i in range(0, len(x), w)]
