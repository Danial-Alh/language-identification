import enum
import os
import random
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from language_identification.utils import load_json, load_pickle, save_json, save_pickle


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def char_tokenizer(x: str):
    return x


def unigram_tokenizer(x: str):
    return x.split()


def subword_tokenizer(x: str, w: int):
    return [x[i : i + w] for i in range(0, len(x), w)]


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
        save_json(self.itos, f"{path}/itos.json")

    @classmethod
    def load(cls, path) -> "ClassEncoder":
        path = path.rstrip("/")
        itos = load_json(f"{path}/itos.json")
        return ClassEncoder(itos=itos)


class ClassifierType(enum.StrEnum):
    NaiveChar = "naive-char"
    NaiveUnigram = "naive-unigram"
    NaiveSubword = "naive-subword"
    MLP = "mlp"


class BaseClassifer:
    def predict(self, sentences: list[str]) -> list[str]:
        pass

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass


class NaiveClassifier(BaseClassifer):
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
        save_npz(f"{path}/token-lang-scores.npz", self.token_lang_scores)

    @classmethod
    def load(cls, path: str) -> "NaiveClassifier":
        path = path.rstrip("/")
        tokenize_fn = load_pickle(f"{path}/tokenize-fn.pkl")
        lang_encoder = ClassEncoder.load(f"{path}/lang-encoder")
        token_encoder = ClassEncoder.load(f"{path}/token-encoder")
        token_lang_scores = load_npz(f"{path}/token-lang-scores.npz")

        return NaiveClassifier(
            tokenize_fn=tokenize_fn,
            lang_encoder=lang_encoder,
            token_encoder=token_encoder,
            token_lang_scores=token_lang_scores,
        )


class MLPClassifier(BaseClassifer):
    def __init__(self, mlp_model, tfidf_model):
        self.mlp_model = mlp_model
        self.tfidf_model = tfidf_model

    def predict(self, sentences: list[str]) -> list[str]:
        X = self.tfidf_model.transform(sentences)
        return self.mlp_model.predict(X)

    def save(self, path: str):
        path = path.rstrip("/")
        os.makedirs(path, exist_ok=True)

        save_pickle(self.mlp_model, f"{path}/mlp-model.pkl")
        save_pickle(self.tfidf_model, f"{path}/tfidf-model.pkl")

    @classmethod
    def load(cls, path: str):
        mlp_model = load_pickle(f"{path}/mlp-model.pkl")
        tfidf_model = load_pickle(f"{path}/tfidf-model.pkl")
        return MLPClassifier(mlp_model=mlp_model, tfidf_model=tfidf_model)


classifier_type2classifier_class: dict[ClassifierType, BaseClassifer] = {
    ClassifierType.NaiveChar: NaiveClassifier,
    ClassifierType.NaiveUnigram: NaiveClassifier,
    ClassifierType.NaiveSubword: NaiveClassifier,
    ClassifierType.MLP: MLPClassifier,
}


class AllInOneClassifier(BaseClassifer):
    def __init__(
        self,
        ordered_classifiers: dict[ClassifierType, BaseClassifer],
        lang2classifier_type: dict[str, ClassifierType],
    ) -> None:
        self.ordered_classifiers = ordered_classifiers
        self.lang2classifier_type = lang2classifier_type

    def predict(self, sentences: list[str]):
        sentences = np.array(sentences)
        sentence_indices = np.arange(len(sentences))

        preds = [None] * len(sentences)
        for clf_type, clf in self.ordered_classifiers.items():
            clf_predictions = np.array(clf.predict(sentences))
            mask = np.array(
                [
                    self.lang2classifier_type[pred] == clf_type
                    for pred in clf_predictions
                ]
            )

            for idx, pred in zip(
                sentence_indices[mask], clf_predictions[mask], strict=True
            ):
                preds[idx] = pred

            sentences = sentences[~mask]
            sentence_indices = sentence_indices[~mask]

            if sentences.shape[0] == 0:
                break
        assert sentences.shape[0] == 0
        assert not any([res is None for res in preds])

        return preds

    def save(self, path: str):
        path = path.rstrip("/")
        os.makedirs(path, exist_ok=True)
        for clf_type, clf in self.ordered_classifiers.items():
            clf.save(f"{path}/{clf_type}")
        save_json(self.lang2classifier_type, f"{path}/langs-classifier.json")
        save_json(
            list(self.ordered_classifiers.keys()), f"{path}/classifier-orders.json"
        )

    @classmethod
    def load(cls, path: str):
        classifier_orders = load_json(f"{path}/classifier-orders.json")
        ordered_classifiers = {}
        for clf_type in classifier_orders:
            clf = classifier_type2classifier_class[clf_type].load(f"{path}/{clf_type}")
            ordered_classifiers[clf_type] = clf
        lang2classifier_type = load_json(f"{path}/langs-classifier.json")
        return AllInOneClassifier(
            ordered_classifiers=ordered_classifiers,
            lang2classifier_type=lang2classifier_type,
        )
