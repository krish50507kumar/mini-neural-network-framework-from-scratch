from models.nn_model import NeuralNetwork
from layers.dense import Dense
from layers.dropout import Dropout
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
    Dropout(0.3),
    Dense(4, 2),
    Softmax()
]
model = NeuralNetwork(layers)

model.compile(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(lr=0.1)
)

model.fit(X, y, epochs=200, batch_size=16, lambda_=0.001,verbose = False)

model.evaluate(X, y)

preds = model.predict(X)
print(preds)