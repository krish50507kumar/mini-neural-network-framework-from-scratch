import numpy as np
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
                if layer.b is not None:
                    layer.b -= self.lr * layer.db
        # for i, layer in enumerate(layers):
        #     if hasattr(layer, "W"):
        #         print(f"Layer {i}")
        #         print("Before:", layer.W[0][0])
        #         print("Grad:", layer.dW[0][0])
        #
        #         layer.W -= self.lr * layer.dW
        #
        #         print("After:", layer.W[0][0])