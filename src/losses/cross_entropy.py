import numpy as np

class CategoricalCrossEntropy:

    def forward(self, y_pred, y_true, layers=None, lambda_=0.0):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        data_loss = -np.sum(y_true * np.log(y_pred), axis=1)
        data_loss = np.mean(data_loss)

        reg_loss = 0.0

        if layers is not None:
            for layer in layers:
                if hasattr(layer, "W"):
                    reg_loss += np.sum(layer.W ** 2)

        reg_loss *= lambda_

        return data_loss + reg_loss

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]