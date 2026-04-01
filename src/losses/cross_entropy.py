import numpy as np

class CategoricalCrossEntropy:

    def forward(self, y_pred, y_true):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]