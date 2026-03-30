import numpy as np

class Dense:
    """
    - it initializes weights and bias with random values using He initialization
    - it has forward(x) propagation
    """
    def __init__(self,input_dim,output_dim,use_bias=True):
        """
        :param input_dim: int
        :param output_dim: int
        :param use_bias: bool defaults (True)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        limit = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim,output_dim).astype(np.float32)*limit
        if use_bias:
            self.b = np.zeros((1,output_dim)).astype(np.float32)
        else:
            self.b = None

    def forward(self,x):
        """
        :param x: np.array
        :return: np.array
        """
        self.x = x
        assert x.shape[1] == self.input_dim, "Input dimension mismatch"
        output = np.dot(x,self.W)
        if self.b is not None:
            output = output + self.b
        return output

    def backward(self,dz):
        """
        :param dz: np.array
        :param lr: int
        :return: np.array

        assume:
        X  = (batch, input_dim)
        W  = (input_dim, output_dim)
        dZ = (batch, output_dim)
        """
        self.dW = np.dot(self.x.T,dz) # shape: (input_dim,output_dim)
        self.db = np.sum(dz,axis=0,keepdims=True) if self.b is not None else None # shape: (1,output_dim)
        dX = np.dot(dz,self.W.T) # shape: (batch,input_dim)
        return dX
