import numpy as np

class MSE:
    def __init__(self):
        pass
    def forward(self, x, y):
        return np.mean(((x - y) ** 2))
    def backward(self, x, y):
        return 2*(x-y)/len(x)