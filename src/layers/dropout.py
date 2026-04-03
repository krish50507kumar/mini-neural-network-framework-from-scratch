import numpy as np

class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, x, training=True):
        if not training:
            return x

        self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)

        return (x * self.mask) / (1.0 - self.rate)

    def backward(self, dZ, lambda_=0.0):
        return (dZ * self.mask) / (1.0 - self.rate)