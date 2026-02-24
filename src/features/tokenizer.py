import os
import json
import tensorflow as tf
from src.config.config import CONFIG


class TextTokenizer:

    def __init__(self):
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=CONFIG["max_vocab"],
            output_mode="int",
            output_sequence_length=CONFIG["max_len"],
            standardize=None
        )

    def fit(self, texts):
        self.vectorizer.adapt(texts)

    def transform(self, texts):
        return self.vectorizer(texts)

    def save(self):
        os.makedirs(CONFIG["tokenizer_dir"], exist_ok=True)

        cfg = self.vectorizer.get_config()
        vocab = self.vectorizer.get_vocabulary()

        with open(os.path.join(CONFIG["tokenizer_dir"], "vectorizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        with open(os.path.join(CONFIG["tokenizer_dir"], "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f)

    def load(self):
        with open(os.path.join(CONFIG["tokenizer_dir"], "vectorizer_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        with open(os.path.join(CONFIG["tokenizer_dir"], "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.vectorizer = tf.keras.layers.TextVectorization.from_config(cfg)
        self.vectorizer.set_vocabulary(vocab)
        return self