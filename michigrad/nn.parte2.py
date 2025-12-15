import random
from michigrad.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class ReLU(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [val.relu() for val in x]
        return x.relu()
    
    def __repr__(self):
        return "ReLU"

class Tanh(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [val.tanh() for val in x]
        return x.tanh()

    def __repr__(self):
        return "Tanh"

class Sigmoid(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [val.sigmoid() for val in x]
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid"

# ----------------------------------------------------

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"LinearNeuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts, activation=ReLU):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            self.layers.append(Layer(sz[i], sz[i+1]))
            if i != len(nouts) - 1:
                self.layers.append(activation())

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"