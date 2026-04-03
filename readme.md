# 🧠 Mini Neural Network Framework (From Scratch)

A lightweight deep learning framework built from scratch using NumPy.
This project implements the full training pipeline — forward pass, backpropagation, optimization, and regularization — without relying on high-level libraries like TensorFlow or PyTorch.

---

## 🚀 What This Project Does

* Builds neural networks layer-by-layer
* Trains using backpropagation
* Supports mini-batch gradient descent
* Implements modern optimizers and regularization
* Provides a clean training API similar to real frameworks

---

## 🏗️ Features

### Core Layers

* Dense (Fully Connected Layer)
* ReLU Activation
* Softmax Activation

### Loss Functions

* Categorical Cross Entropy

### Optimizers

* SGD
* Adam (Adaptive Moment Estimation)

### Regularization

* L2 Regularization
* Dropout

### Training System

* Forward propagation
* Backward propagation
* Mini-batch training
* Accuracy evaluation

### API Design

```python
model.compile(...)
model.fit(...)
model.evaluate(...)
model.predict(...)
```

### Utilities

* Model saving & loading (pickle-based)
* One-hot encoding support

---

## 📁 Project Structure

```
mini-neural-network-framework-from-scratch/
├── Models/
│   ├── iris_model.pkl        # Saved weights for Iris classification
│   └── xor_model.pkl         # Saved weights for XOR problem
├── src/
│   ├── __init__.py           # Framework versioning (v1.0.0)
│   ├── activations/
│   │   ├── relu.py           # ReLU activation implementation
│   │   ├── sigmoid.py        # Sigmoid activation implementation
│   │   └── softmax.py        # Softmax for multi-class output
│   ├── core/
│   │   └── module.py         # Base parameter tracking
│   ├── evaluation/
│   │   └── metrics.py        # Accuracy calculation logic
│   ├── layers/
│   │   ├── dense.py          # Fully connected layer with He initialization
│   │   └── dropout.py        # Dropout regularization layer
│   ├── losses/
│   │   ├── cross_entropy.py  # Categorical Cross-Entropy with L2 reg
│   │   └── mse.py            # Mean Squared Error implementation
│   ├── models/
│   │   ├── base_model.py     # Basic forward/backward wrapper
│   │   └── nn_model.py       # Main NeuralNetwork class with fit/save/load
│   └── optimizers/
│       ├── adam.py           # Adam optimizer with momentum/scaling
│       └── sgd.py            # Stochastic Gradient Descent
├── tests/
│   └── test_1.py             # Iris dataset training script
├── .gitignore                # Rules to skip venv and IDE files
├── LICENSE                   # MIT License
├── main.py                   # XOR training example
└── README.md                 # Project documentation and usage guide
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/mini-neural-network-framework-from-scratch.git
cd mini-neural-network-framework-from-scratch
pip install numpy scikit-learn
```

---

## 🧪 Example Usage

```python
from models.nn_model import NeuralNetwork
from layers.dense import Dense
from layers.dropout import Dropout
from activations.relu import ReLU
from activations.softmax import Softmax
from losses.cross_entropy import CategoricalCrossEntropy
from evaluation.metrics import accuracy
from optimizers.adam import Adam
import numpy as np

model = NeuralNetwork([
    Dense(4, 8),
    ReLU(),
    Dropout(0.3),
    Dense(8, 3),
    Softmax()
])

model.compile(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(lr=0.01)
)

model.fit(X_train, y_train, epochs=200, batch_size=16, lambda_=0.001)

model.evaluate(X_test, y_test)
```

---

## 📊 Results

### Iris Dataset

* Training Accuracy: ~95–100%
* Test Accuracy: ~90–100%

---

## 🧠 Key Learnings

* Implemented backpropagation from scratch
* Understood gradient flow across layers
* Built modular and extensible architecture
* Explored effects of regularization (L2, Dropout)
* Designed a clean training API similar to real-world frameworks

---

## ⚠️ Limitations

* CPU-only (NumPy-based)
* No GPU acceleration
* No automatic differentiation
* Limited to basic architectures

---

## 📜 License

MIT License

---

## 👤 Author

Krish

---

## ⭐ If you found this useful

Give it a star and build something even better.
