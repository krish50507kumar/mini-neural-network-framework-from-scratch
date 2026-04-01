import numpy as np

class Sigmoid:
    def forward(self,x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self,dZ,lambda_ = 0.0):
        s = 1/(1+ np.exp(-self.x))
        return dZ * s * (1-s)