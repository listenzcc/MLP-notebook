# Experiment of MLP

[MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron "MLP")
: Multilayer perception,

>A multilayer perceptron (MLP) is a fully connected class of feedforward artificial neural network (ANN). The term MLP is used ambiguously, sometimes loosely to mean any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation); see § Terminology. Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.[1]

I will make some experiment about it.

---

## Experiment-1

The experiment is performed to test the edge of the MLP.

The spiral shaped point clouds are built,
and the MLP is used to predict one from another.

Currently, in 20230101, the MLP estimates at a quite low accuracy.
The experiment code is provided in
[experiment-1/main.py](./experiment-1/main.py).

The MLP is built by torchvision,
and the learning process is defined as following.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torchvision.ops.MLP(2, [4, 3, 2], activation_layer=nn.LeakyReLU)
        self.sig = nn.Tanh()

    def forward(self, x):
        return self.sig(self.mlp(x))


net = Net().cuda()

lr = 1e-2
optimizer = torch.optim.AdamW(net.parameters(), lr)
criterion = nn.MSELoss()
```

The model is trained for 1000 times,
and the outcome ends at

| Step  | Loss(train) | Loss(test) |
| :---: | :---------: | :--------: |
| 0960  |   0.0280    |   0.0251   |
| 0970  |   0.0213    |   0.0251   |
| 0980  |   0.0223    |   0.0251   |
| 0990  |   0.0234    |   0.0254   |

The dataset of the point clouds are presented in following,
it contains the 2D and 3D versions,
and the third dim of the 3D version are the order of the points.

![dataset.png](experiment-1/doc/dataset.png)

![dataset-1.png](experiment-1/doc/dataset-1.png)

The outcome shows the prediction,
with a low accuracy.
The 2D and 3D versions are also provided.

![pred.png](experiment-1/doc/pred.png)

![pred-1.png](experiment-1/doc/pred-1.png)

<div>

<iframe src='./experiment-1/doc/spiral-1.html' style='width: 800px; height: 800px'></iframe>

<iframe src='./experiment-1/doc/spiral-2.html' style='width: 800px; height: 800px'></iframe>

<div>