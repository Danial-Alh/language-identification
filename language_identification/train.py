# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..
# %%
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from tqdm.auto import trange

from language_identification.models import (
    AllInOneClassifier,
    ClassEncoder,
    ClassifierType,
)
from language_identification.models import MLPClassifier as BundeledMLPClassifier
from language_identification.models import (
    NaiveClassifier,
    char_tokenizer,
    set_seed,
    subword_tokenizer,
    unigram_tokenizer,
)

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

print(f"{len(char_set)=}")

# %%

lang_encoder = ClassEncoder(tr_df["label_str"])
lang_classifiers = {}
acceptable_f1 = 0.8
seed = 1324809
set_seed(seed=seed)

# %%

char_model = NaiveClassifier(tokenize_fn=char_tokenizer, lang_encoder=lang_encoder)
char_model.fit(tr_df)

iloc = 0
print(
    "char predictor test:",
    char_model.predict([tr_df.iloc[iloc]["sentence"]]),
    "label:",
    tr_df.iloc[iloc]["label_str"],
)

# %%

W = 1000
preds = [
    res
    for i in trange(0, ts_df.shape[0], W)
    for res in char_model.predict(ts_df["sentence"].values[i : i + W])
]

report = classification_report(ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(ts_df["label_str"], preds, output_dict=True)
).T
report = report[report.index.isin(lang_encoder.stoi)]

# %%

plt.hist(report["f1-score"])
plt.show(block=False)

print(
    "lanuguages detected with chars:",
    (report["f1-score"].sort_values(ascending=False) >= acceptable_f1).sum(),
)

for lang in report[report["f1-score"] >= acceptable_f1].index:
    lang_classifiers[lang] = ClassifierType.NaiveChar

# %%
remaining_langs = set(report[report["f1-score"] <= 0.8].index.values)
curr_tr_df = tr_df[tr_df["label_str"].isin(remaining_langs)]
curr_ts_df = ts_df[ts_df["label_str"].isin(remaining_langs)]

# %%

unigram_model = NaiveClassifier(unigram_tokenizer, lang_encoder)
unigram_model.fit(curr_tr_df)

unigram_model.predict([curr_tr_df.iloc[0]["sentence"]])  # test

# %%

W = 1000
preds = [
    res
    for i in trange(0, curr_ts_df.shape[0], W)
    for res in unigram_model.predict(curr_ts_df["sentence"].values[i : i + W])
]

report = classification_report(curr_ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(curr_ts_df["label_str"], preds, output_dict=True)
).T
report = report[report.index.isin(lang_encoder.stoi)]

# %%
plt.hist(report["f1-score"])
plt.show(block=False)

print(
    "lanuguages detected with unigrams:",
    (report["f1-score"].sort_values(ascending=False) >= acceptable_f1).sum(),
)


for lang in report[report["f1-score"] >= acceptable_f1].index:
    lang_classifiers[lang] = ClassifierType.NaiveUnigram

# %%
remaining_langs = set(report[report["f1-score"] <= 0.8].index.values)
curr_tr_df = tr_df[tr_df["label_str"].isin(remaining_langs)]
curr_ts_df = ts_df[ts_df["label_str"].isin(remaining_langs)]
# %%

subword_len = 3

subword_model = NaiveClassifier(
    tokenize_fn=partial(subword_tokenizer, w=subword_len), lang_encoder=lang_encoder
)
subword_model.fit(curr_tr_df)

subword_model.predict([curr_tr_df.iloc[0]["sentence"]])  # test

# %%
W = 1000
preds = [
    res
    for i in trange(0, curr_ts_df.shape[0], W)
    for res in subword_model.predict(curr_ts_df["sentence"].values[i : i + W])
]

report = classification_report(curr_ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(curr_ts_df["label_str"], preds, output_dict=True)
).T
report = report[report.index.isin(lang_encoder.stoi)]

print(
    "lanuguages detected with subwords:",
    (report["f1-score"].sort_values(ascending=False) >= acceptable_f1).sum(),
)

for lang in report[report["f1-score"] >= acceptable_f1].index:
    lang_classifiers[lang] = ClassifierType.NaiveSubword

# %%
remaining_langs = set(report[report["f1-score"] < acceptable_f1].index.values)

mask = [
    preds[i] in remaining_langs or curr_ts_df["label_str"].iloc[i] in remaining_langs
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
plt.show(block=False)

# %%
curr_tr_df = tr_df[tr_df["label_str"].isin(remaining_langs)]
curr_ts_df = ts_df[ts_df["label_str"].isin(remaining_langs)]

# %%

tfidf_model = TfidfVectorizer(
    lowercase=True,
    min_df=1,
    use_idf=True,
    ngram_range=(1, 1),
)
X = tfidf_model.fit_transform(curr_tr_df["sentence"])

print(X.shape)

logistic_model = LogisticRegression(C=100)
logistic_model.fit(X, curr_tr_df["label_str"])

print("predicting")
X_ts = tfidf_model.transform(curr_ts_df["sentence"])
preds = logistic_model.predict(X_ts)

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
mlp_model = BundeledMLPClassifier(mlp_model=mlp_model, tfidf_model=tfidf_model)


print("predicting")
preds = mlp_model.predict(curr_ts_df["sentence"])

report = classification_report(curr_ts_df["label_str"], preds)
print(report)
report = pd.DataFrame(
    classification_report(curr_ts_df["label_str"], preds, output_dict=True)
).T
report = report[report.index.isin(lang_encoder.stoi)]

print("lanuguages detected with mlp classifier:", report.shape[0])

for lang in report.index:
    lang_classifiers[lang] = ClassifierType.MLP

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
plt.show(block=False)

# %%

print(curr_ts_df[(preds == "bos") & (curr_ts_df["label_str"] == "bos")])

# I checked some 'hbs' and 'hrv' languages in the google translate. all of them were crotian. maybe it is also hard for google to detect it they are same langs and it is the problem of the datset
#### remove puncs

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
all_in_one_classifer = AllInOneClassifier(
    ordered_classifiers={
        ClassifierType.NaiveChar: char_model,
        ClassifierType.NaiveUnigram: unigram_model,
        ClassifierType.NaiveSubword: subword_model,
        ClassifierType.MLP: mlp_model,
    },
    lang2classifier_type=lang_classifiers,
)

all_in_one_classifer.save("data/final-model")


all_in_one_classifer = AllInOneClassifier.load("data/final-model")

# %%
W = 1000
preds = [
    res
    for i in trange(0, ts_df.shape[0], W)
    for res in all_in_one_classifer.predict(ts_df["sentence"].values[i : i + W])
]

report = classification_report(ts_df["label_str"], preds)
print(report)

# %%
