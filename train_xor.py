from michigrad.engine import Value
from michigrad.nn import Layer
from michigrad.visualize import show_graph

xs = [
    [Value(0), Value(0)],
    [Value(0), Value(1)],
    [Value(1), Value(0)],
    [Value(1), Value(1)],
]

ys = [0, 1, 1, 0]

model = Layer(2, 2, nonlin=False)

def forward_dataset():
    loss = Value(0.0, name="loss")

    for x, y in zip(xs, ys):
        out = model(x)
        y_pred = out[0] + out[1]
        y_true = Value(y)

        err = (y_pred - y_true) ** 2
        loss = loss + err

    return loss

def forward_single(x, y):
    out = model(x)
    y_pred = out[0] + out[1]
    y_true = Value(y)

    loss = (y_pred - y_true) ** 2
    loss.name = "loss"

    return loss

x_vis = [Value(1), Value(0)]
y_vis = 1

loss_vis = forward_single(x_vis, y_vis)
print("Loss (single example, forward):", loss_vis.data)

show_graph(loss_vis)


model.zero_grad()
loss_vis.backward()


show_graph(loss_vis)

def train(epochs=20, lr=0.01):
    history = []

    for epoch in range(epochs):
        loss = forward_dataset()

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data += -lr * p.grad

        history.append(loss.data)
        print(f"Iteración {epoch:02d} | loss = {loss.data:.4f}")

    return history

train()

