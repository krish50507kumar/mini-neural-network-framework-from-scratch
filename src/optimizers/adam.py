import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1

        for i, layer in enumerate(layers):

            if hasattr(layer, "W"):

                if i not in self.m:
                    self.m[i] = {"dW": np.zeros_like(layer.W),
                                 "db": np.zeros_like(layer.b)}
                    self.v[i] = {"dW": np.zeros_like(layer.W),
                                 "db": np.zeros_like(layer.b)}

                self.m[i]["dW"] = self.beta1 * self.m[i]["dW"] + (1 - self.beta1) * layer.dW
                self.m[i]["db"] = self.beta1 * self.m[i]["db"] + (1 - self.beta1) * layer.db

                self.v[i]["dW"] = self.beta2 * self.v[i]["dW"] + (1 - self.beta2) * (layer.dW ** 2)
                self.v[i]["db"] = self.beta2 * self.v[i]["db"] + (1 - self.beta2) * (layer.db ** 2)

                m_hat_dW = self.m[i]["dW"] / (1 - self.beta1 ** self.t)
                m_hat_db = self.m[i]["db"] / (1 - self.beta1 ** self.t)

                v_hat_dW = self.v[i]["dW"] / (1 - self.beta2 ** self.t)
                v_hat_db = self.v[i]["db"] / (1 - self.beta2 ** self.t)

                layer.W -= self.lr * m_hat_dW / (np.sqrt(v_hat_dW) + self.epsilon)
                layer.b -= self.lr * m_hat_db / (np.sqrt(v_hat_db) + self.epsilon)