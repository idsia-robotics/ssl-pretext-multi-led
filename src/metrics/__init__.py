import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

def binary_auc(preds, trues):
    fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
    return metrics.auc(fpr, tpr)

def mse(preds, trues):
    return np.mean((trues - preds) ** 2)


def angle_difference(preds, trues):

    def custom_modulo(a, b):
        return a - np.floor(a / b) * b
    return np.abs(custom_modulo(trues - preds + np.pi, np.pi * 2) - np.pi)


def leds_auc(led_preds, led_trues):
    total_score = 0
    scores = []
    for i in range(led_preds.shape[1]):
        score = binary_auc(led_preds[:, i], led_trues[:, i])
        scores.append(score)
    total_score = sum(scores) / len(scores)
    return total_score, scores
