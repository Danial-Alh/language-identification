import json
import os
import pickle


def save_pickle(obj, filepath):
    folderpath = "/".join(filepath.split("/")[:-1])
    if folderpath:
        os.makedirs(folderpath, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(obj, filepath):
    folderpath = "/".join(filepath.split("/")[:-1])
    if folderpath:
        os.makedirs(folderpath, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f)


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)
