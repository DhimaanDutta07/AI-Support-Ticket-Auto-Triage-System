import os
import json


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)