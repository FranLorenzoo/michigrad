from michigrad.engine import Value
from michigrad.visualize import show_graph
from michigrad.nn_parte2 import MLP, Layer, ReLU, Module
import random

# Datos xor
xs = [
    [Value(0), Value(0)],
    [Value(0), Value(1)],
    [Value(1), Value(0)],
    [Value(1), Value(1)]
]

ys = [
    Value(0),
    Value(1),
    Value(1),
    Value(0)
]

# Creacion de red con 2 capas y utilizando funcion de activacion ReLU
model = MLP(2, [2, 1], activation=ReLU)


def forward_dataset():
    loss = Value(0.0, name="loss")

    for x, y in zip(xs, ys):
        y_pred = model(x) 
        err = (y_pred - y) ** 2
        loss = loss + err

    return loss

def forward_single(x, y):
    y_pred = model(x)        
    loss = (y_pred - y) ** 2
    loss.name = "loss"
    
    return loss, y_pred
 
 # Entrenamiento
def train(epochs=100, lr=0.1):
    history = []
    for epoch in range(epochs):
        loss = forward_dataset()
        
        model.zero_grad()
        loss.backward()
        
        for p in model.parameters():
            p.data -= lr * p.grad

        history.append(loss.data)
        print(f"Iteración {epoch:02d} | loss = {loss.data:.4f}")

    return history
