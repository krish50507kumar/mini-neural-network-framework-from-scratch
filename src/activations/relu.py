import numpy as np
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dZ,lambda_ = 0.0):
        return dZ * (self.x > 0)