# %%
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm, trange

# %%

hf_ds = load_dataset("MartinThoma/wili_2018")

tr_df = hf_ds["train"].to_pandas()
ts_df = hf_ds["test"].to_pandas()
# %%

tr_df["label_str"] = tr_df["label"].apply(
    lambda x: hf_ds["train"].info.features["label"].int2str(x)
)
ts_df["label_str"] = ts_df["label"].apply(
    lambda x: hf_ds["test"].info.features["label"].int2str(x)
)
# %%

char_set = set()
for x in tr_df["sentence"]:
    for xx in x:
        char_set.add(xx)

print(f"{len(char_set)}")
# %%

char2langlist = defaultdict(list)
for rec in tr_df.itertuples():
    for x in rec.sentence:
        char2langlist[x].append(rec.label_str)

# %%
char2langdist = {c: Counter(v) for c, v in char2langlist.items()}

# %%


class ClassEncoder:
    def __init__(self, values=None, itos=None) -> None:
        assert not (values is None and itos is None)
        self.itos = list(sorted(set(values))) if itos is None else list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


char_encoder = ClassEncoder(char2langdist.keys())
lang_encoder = ClassEncoder(tr_df["label_str"])

# %%
char_lang_scores = []
for c, langdist in char2langdist.items():
    total_lang_score = np.sum(langdist.values())
    for lang, score in langdist.items():
        char_lang_scores.append((score, char_encoder.stoi[c], lang_encoder.stoi[lang]))

d, r, c = zip(*char_lang_scores)
char_lang_scores = csr_matrix(
    (d, (r, c)),
    shape=(len(char_encoder), len(lang_encoder)),
)
# %%


def predict_character_naive(sentences):
    sentences_vec = []
    for i, sentence in enumerate(sentences):
        for c, in_sentence_freq in Counter([c for c in sentence if c.strip()]).items():
            if c not in char_encoder.stoi:
                continue
            sentences_vec.append((in_sentence_freq, i, char_encoder.stoi[c]))
    d, r, c = zip(*sentences_vec)
    sentences_vec = csr_matrix((d, (r, c)), shape=(len(sentences), len(char_encoder)))
    probs = (sentences_vec @ char_lang_scores).toarray()
    argmax_indices = np.argmax(probs, axis=-1)
    return [lang_encoder.itos[idx] for idx in argmax_indices]


iloc = 0
print(
    predict_character_naive([tr_df.iloc[iloc]["sentence"]]),
    tr_df.iloc[iloc]["label_str"],
)

# %%

W = 1000
preds = [
    res
    for i in trange(0, ts_df.shape[0], W)
    for res in predict_character_naive(ts_df["sentence"].values[i : i + W])
]

report = classification_report(ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(ts_df["label_str"], preds, output_dict=True)
).T

# %%

plt.hist(report["f1-score"])
plt.show()

print(
    "lanuguages detected with chars:",
    (report["f1-score"].sort_values(ascending=False) > 0.8).sum(),
)
# %%
remaining_langs = set(report[report["f1-score"] <= 0.8].index.values)
curr_tr_df = tr_df[tr_df["label_str"].isin(remaining_langs)]
curr_ts_df = ts_df[ts_df["label_str"].isin(remaining_langs)]
# %%


def build_token2langs(df, tokenize_fn):
    token2langs = defaultdict(list)
    for rec in df.itertuples():
        for x in tokenize_fn(rec.sentence):
            token2langs[x].append(rec.label_str)
    return token2langs


def build_token_lang_similarity_matrix(token2langs, token_encoder: ClassEncoder):
    token_lang_scores = []
    for token, langs in token2langs.items():
        for lang, score in Counter(langs).items():
            score /= len(langs)
            token_lang_scores.append(
                (score, token_encoder.stoi[token], lang_encoder.stoi[lang])
            )

    d, r, c = zip(*token_lang_scores)
    token_lang_scores = csr_matrix(
        (d, (r, c)),
        shape=(len(token2langs), len(lang_encoder)),
    )

    return token_lang_scores


def build_naive_predictor(tokenizer_fn, token_lang_scores, token_encoder: ClassEncoder):
    def _predictor(sentences):
        sentences_vec = []
        for i, sentence in enumerate(sentences):
            for c, in_sentence_freq in Counter(
                [c for c in tokenizer_fn(sentence) if c.strip()]
            ).items():
                if c not in token_encoder.stoi:
                    continue
                sentences_vec.append((in_sentence_freq, i, token_encoder.stoi[c]))
        d, r, c = zip(*sentences_vec)
        sentences_vec = csr_matrix(
            (d, (r, c)), shape=(len(sentences), len(token_encoder))
        )
        probs = (sentences_vec @ token_lang_scores).toarray()
        argmax_indices = np.argmax(probs, axis=-1)
        return [lang_encoder.itos[idx] for idx in argmax_indices]

    return _predictor


# %%

unigram_tokenizer = lambda x: x.split()
unigram2langs = build_token2langs(curr_tr_df, unigram_tokenizer)
unigram_encoder = ClassEncoder(unigram2langs.keys())
unigram_lang_scores = build_token_lang_similarity_matrix(unigram2langs, unigram_encoder)
unigram_predictor = build_naive_predictor(
    unigram_tokenizer, unigram_lang_scores, unigram_encoder
)

print(unigram_predictor([curr_tr_df.iloc[0]["sentence"]]))

# %%

W = 1000
preds = [
    res
    for i in trange(0, curr_ts_df.shape[0], W)
    for res in unigram_predictor(curr_ts_df["sentence"].values[i : i + W])
]

report = classification_report(curr_ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(curr_ts_df["label_str"], preds, output_dict=True)
).T

# %%
plt.hist(report["f1-score"])
plt.show()

print(
    "lanuguages detected with chars:",
    (report["f1-score"].sort_values(ascending=False) > 0.8).sum(),
)
# %%
remaining_langs = set(report[report["f1-score"] <= 0.8].index.values)
curr_tr_df = tr_df[tr_df["label_str"].isin(remaining_langs)]
curr_ts_df = ts_df[ts_df["label_str"].isin(remaining_langs)]
# %%

subword_len = 3
subword_tokenizer = lambda x: [
    x[i : i + subword_len] for i in range(0, len(x), subword_len)
]
subword2langs = build_token2langs(curr_tr_df, subword_tokenizer)
subword_encoder = ClassEncoder(subword2langs.keys())
subword_lang_scores = build_token_lang_similarity_matrix(subword2langs, subword_encoder)
subword_predictor = build_naive_predictor(
    subword_tokenizer, subword_lang_scores, subword_encoder
)

print(subword_predictor([curr_tr_df.iloc[0]["sentence"]]))

# %%
W = 1000
preds = [
    res
    for i in trange(0, curr_ts_df.shape[0], W)
    for res in subword_predictor(curr_ts_df["sentence"].values[i : i + W])
]

report = classification_report(curr_ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(curr_ts_df["label_str"], preds, output_dict=True)
).T

# %%
selected_langs = ["bos", "hbs", "hrv"]

mask = [
    preds[i] in selected_langs or curr_ts_df["label_str"].iloc[i] in selected_langs
    for i in range(len(preds))
]

preds = np.array(preds)


print(classification_report(curr_ts_df.loc[mask]["label_str"], preds[mask]))
# %%

temp_labels = sorted(
    set(list(curr_ts_df.loc[mask]["label_str"].unique()) + list(preds[mask]))
)
conf_matrix = confusion_matrix(
    curr_ts_df.loc[mask]["label_str"], preds[mask], labels=temp_labels
)

fig, ax = plt.subplots()
ax.imshow(conf_matrix)
ax.set_xticks(np.arange(len(temp_labels)), labels=temp_labels)  # pred
ax.set_yticks(np.arange(len(temp_labels)), labels=temp_labels)  # true

for i in range(len(temp_labels)):
    for j in range(len(temp_labels)):
        text = ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="w")
plt.show()

# %%
curr_tr_df = tr_df[tr_df["label_str"].isin(selected_langs)]
curr_ts_df = ts_df[ts_df["label_str"].isin(selected_langs)]

# %%

tfidf_model = TfidfVectorizer(
    lowercase=True,
    min_df=1,
    use_idf=True,
    ngram_range=(1, 1),
)
X = tfidf_model.fit_transform(curr_tr_df["sentence"])

print(X.shape)

model = LogisticRegression(C=100)
model.fit(X, curr_tr_df["label_str"])

print("predicting")
X_ts = tfidf_model.transform(curr_ts_df["sentence"])
preds = model.predict(X_ts)

report = classification_report(curr_ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(curr_ts_df["label_str"], preds, output_dict=True)
).T
# %%

mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation="relu",
    learning_rate_init=1e-3,
)

mlp_model.fit(X, curr_tr_df["label_str"])

print("predicting")
preds = mlp_model.predict(X_ts)

report = classification_report(curr_ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(curr_ts_df["label_str"], preds, output_dict=True)
).T

# %%

temp_labels = sorted(set(list(curr_ts_df["label_str"].unique()) + list(preds)))
conf_matrix = confusion_matrix(curr_ts_df["label_str"], preds, labels=temp_labels)

fig, ax = plt.subplots()
ax.imshow(conf_matrix)
ax.set_xticks(np.arange(len(temp_labels)), labels=temp_labels)  # pred
ax.set_yticks(np.arange(len(temp_labels)), labels=temp_labels)  # true

for i in range(len(temp_labels)):
    for j in range(len(temp_labels)):
        text = ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="w")
plt.show()
# %%

print(curr_ts_df[(preds == "bos") & (curr_ts_df["label_str"] == "bos")])

# I checked some 'hbs' and 'hrv' languages in the google translate. all of them were crotian. maybe it is also hard for google to detect it they are same langs and it is the problem of the datset

# bos but google detected crotians:
# U periodu do 9. vijeka brojna mala kraljevstva...
# Rodio se u Zürichu, Švicarska gdje je na ETH p...
# Prva procjena Michaela Todda za proračun filma...
# Američka mornarica je tako razvila i posebnu...
# Terorizam je smišljena upotreba nezakonitog...
# Interni monolog: kada glumac priča kao da se o...

# bos but correctly predicted and google disagreed:
# Ovo sve ukazuje na to da je velika mogučnost ...
# Suprotno uvriježenom mišljenju, velika većina ...
# Košljoribe su su primitivna ektotermne (hladnokrvne), ...

# google agreed on bos:
# Modernizovana verzija Leopard 2 tenka, modernizovana ...
# %%
