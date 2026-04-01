from models.nn_model import NeuralNetwork
from layers.dense import Dense
from activations.relu import ReLU
from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from optimizers.sgd import SGD
from losses.mse import MSE
from losses.cross_entropy import CategoricalCrossEntropy
from evaluation.metrics import accuracy
from optimizers.adam import Adam
import numpy as np

# input
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
],dtype = np.float32)

y = np.array([
    [0,1],
    [1,0],
    [1,0],
    [0,1]
],dtype = np.float32)

layers = [
    Dense(2, 4),
    ReLU(),
    Dense(4, 2),
    Softmax()
]
model = NeuralNetwork(layers)
model.compile(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(lr = 0.2)
)
model.train(X,y,epochs=100)
# print("Before:", layers[0].W[0][0])
y_pred = model.predict(X)
print(y_pred)
# print("After:", layers[0].W[0][0])
# print(accuracy(y_pred,y))
# endpoint = "Models/"
# path = "xor_model.pkl"
# model.save(endpoint+path)
# print("done")