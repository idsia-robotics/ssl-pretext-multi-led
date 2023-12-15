import numpy as np
from sklearn import metrics

def binary_auc(preds, trues):
    fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
    return metrics.auc(fpr, tpr)

def mse(preds, trues):
    return np.mean((trues - preds) ** 2)


def angle_difference(preds, trues):

    def custom_modulo(a, b):
        return a - np.floor(a / b) * b
    return np.abs(custom_modulo(trues - preds + np.pi, np.pi * 2) - np.pi)