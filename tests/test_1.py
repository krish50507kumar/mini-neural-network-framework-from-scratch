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
from layers.dropout import Dropout
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y = one_hot(y, 3)

X = (X - X.mean(axis=0)) / X.std(axis=0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = NeuralNetwork([
    Dense(4, 8),
    ReLU(),
    Dropout(0.5),
    Dense(8, 3),
    Softmax()
])

model.compile(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(lr=0.05)
)

model.compile(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(lr=0.01)
)

model.fit(X_train, y_train, epochs=200, batch_size=16, lambda_=0.001,verbose=False)
model.evaluate(X_test, y_test)

preds = model.predict(X_test)
print(preds)