import numpy as np
import pickle
from evaluation.metrics import accuracy
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X, training=True):
        for layer in self.layers:
            if hasattr(layer, "forward"):
                if "Dropout" in layer.__class__.__name__:
                    X = layer.forward(X, training)
                else:
                    X = layer.forward(X)
        return X

    def backward(self, dZ,lambda_ = 0.0):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ,lambda_)

    def compile(self,loss,optimizer):
        self.loss_fn = loss
        self.optimizer = optimizer

    def fit(self,X,y,epochs = 10,batch_size=32,lambda_=0.001,verbose = True):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            all_preds = []
            all_true = []
            total_loss = 0
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            for i in range(0,n_samples,batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                # I pushed the input_X or train_X through my neural network and predicted y_pred i,e output of last neuron
                y_pred = self.forward(X_batch, training=True)
                # I now compute the loss of last neuron cause previous layer neurons need it
                loss = self.loss_fn.forward(y_pred,y_batch,lambda_=lambda_)
                total_loss += loss
                # now i compute the gradient of last neuron
                dZ = self.loss_fn.backward(y_pred,y_batch)
                # now we backpropagation the network and update every neuron weight and bias as per their loss
                self.backward(dZ,lambda_)
                self.optimizer.step(self.layers)
                all_preds.append(y_pred)
                all_true.append(y_batch)
            y_pred_full = np.vstack(all_preds)
            y_true_full = np.vstack(all_true)

            epoch_acc = accuracy(y_pred_full, y_true_full)
            batches = int(np.ceil(n_samples / batch_size))
            epoch_loss = total_loss / batches

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    def predict(self, X):
        return self.forward(X,training = True)

    def save(self, path):
        params = []

        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append({
                    "W": layer.W,
                    "b": layer.b
                })

        with open(path, "wb") as f:
            pickle.dump(params, f)

    def load(self, path):
        with open(path, "rb") as f:
            params = pickle.load(f)

        layer_idx = 0

        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W = params[layer_idx]["W"]
                layer.b = params[layer_idx]["b"]
                layer_idx += 1

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        loss = self.loss_fn.forward(y_pred, y, self.layers, lambda_=0.0)
        acc = accuracy(y_pred, y)

        print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return loss, acc