import numpy as np

def accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)

    return np.mean(pred_labels == true_labels)