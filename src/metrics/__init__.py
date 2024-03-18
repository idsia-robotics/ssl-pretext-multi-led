import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

def binary_auc(preds, trues, return_optimal_threshold = False):
    fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
    if not return_optimal_threshold:
        return metrics.auc(fpr, tpr)
    else:
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return metrics.auc(fpr, tpr), optimal_threshold


def mse(preds, trues):
    return np.mean((trues - preds) ** 2)


def angle_difference(preds, trues):

    def custom_modulo(a, b):
        return a - np.floor(a / b) * b
    return np.abs(custom_modulo(trues - preds + np.pi, np.pi * 2) - np.pi)


def leds_auc(led_preds, led_trues, led_visibility= None):
    total_score = 0
    scores = []
    if led_visibility is None:
        led_visibility = np.ones_like(led_preds, dtype=np.int64)

    for i in range(led_preds.shape[1]):
        score = binary_auc(led_preds[led_visibility[:, i], i], led_trues[led_visibility[:, i], i])
        scores.append(score)
    total_score = sum(scores) / len(scores)
    return total_score, scores


def mean_dummy_predictor_error(data, error = 'absolute'):
    mean = data.mean(-1)
    if error == 'absolute':
        error_fn = np.abs
    elif error == 'squared':
        error_fn = lambda x: np.linalg.norm(x, ord = 2, axis = -1)
    
    return error_fn(data - mean)
