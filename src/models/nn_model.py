import numpy as np
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

    def complie(self,loss,optimizer):
        self.loss_fn = loss
        self.optimizer = optimizer

    def train(self,X,y,epochs = 100):
        for epoch in range(epochs):
            # I pushed the input_X or train_X through my neural network and predicted y_pred i,e output of last neuron
            y_pred = self.forward(X)
            # I now compute the loss of last neuron cause previous layer neurons need it
            loss = self.loss_fn.forward(y_pred,y)
            # now i compute the gradient of last neuron
            dZ = self.loss_fn.backward(y_pred,y)
            # now we backpropagation the network and update every neuron weight and bias as per their loss
            self.backward(dZ)
            self.optimizer.step(self.layers)
            print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)