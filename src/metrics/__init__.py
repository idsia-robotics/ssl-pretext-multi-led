from sklearn import metrics

def binary_auc(preds, trues):
    fpr, tpr, thresholds = metrics.roc_curve(trues, preds)
    return metrics.auc(fpr, tpr)
