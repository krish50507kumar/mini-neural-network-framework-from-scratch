from models.nn_model import NeuralNetwork
from layers.dense import Dense
from activations.relu import ReLU
from activations.sigmoid import Sigmoid
from optimizers.sgd import SGD
from losses.mse import MSE
import numpy as np

# input
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
],dtype = np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
],dtype = np.float32)

layers = [
    Dense(2, 4),
    ReLU(),
    Dense(4, 1),
    Sigmoid()
]
model = NeuralNetwork(layers)
model.complie(
    loss=MSE(),
    optimizer=SGD(lr = 0.5)
)
model.train(X,y,epochs =500)
# print("Before:", layers[0].W[0][0])
print(model.predict(X))
# print("After:", layers[0].W[0][0])