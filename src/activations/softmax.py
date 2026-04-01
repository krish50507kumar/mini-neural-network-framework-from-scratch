import numpy as np

class Softmax:

    def forward(self,x):
        self.x = x
        stable_e = np.exp(x - np.max(x,axis=1,keepdims=True))
        self.out = stable_e/np.sum(stable_e,axis=1,keepdims=True)
        return self.out

    def backward(self,dZ):
        return dZ