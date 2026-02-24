import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_prob):
    return float(f1_score(y_true, np.argmax(y_prob, axis=1), average="macro"))