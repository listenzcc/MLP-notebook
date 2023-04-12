
'''
File: main-6.py
'''

# %%
import sys
import time
import torch
import torchvision
import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm


# %%
DEBUG = True
DEBUG = 'ipykernel_launcher.py' in sys.argv[0]

if DEBUG:
    loops = 20
else:
    loops = 20000

# %%
num_features = 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torchvision.ops.MLP(
            num_features, [4, 10, 4, 2], activation_layer=nn.LeakyReLU)
        self.act = nn.Tanh()

    def forward(self, x):
        output = self.mlp(x)
        output = self.act(output)
        return output


net = Net().cuda()

# %%

# lr = 1e-2
# optimizer = torch.optim.AdamW(net.parameters(), lr)
optimizer = torch.optim.AdamW(net.parameters())
criterion = nn.MSELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.MSELoss()
net

# %%
samples = 10000

lst = []

for j in range(samples):
    p = np.random.random((10, 1))
    p /= np.sum(p)

    w = np.random.random((10, 1))
    w /= np.sum(p * w)

    pw = np.concatenate([p, w], axis=1)
    lst.append(pw)

data = np.array(lst)
data.shape


# %%
loops = 10000
alpha = 1.1

losses = []

cy = torch.Tensor([alpha]).cuda()
cy2 = torch.Tensor([1.00]).cuda()

for j in tqdm(range(loops)):
    np.random.shuffle(data)
    x = torch.Tensor(data[0]).cuda()
    y = net(x)
    d = y[:, 0].matmul(y[:, 1])
    d2 = y[:, 0].sum()

    l1 = criterion(d, cy)
    l2 = criterion2(d2, cy2)
    l3 = criterion3(y[:, [1]], x[:, [1]])
    loss = l1 + 0.5 * l2 + 0.05 * l3

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    v = (l1.item(), l2.item(), l3.item(), loss.item())
    print(v)
    losses.append(v)

losses

# %%
plt.plot(losses)
plt.legend(['loss: growth', 'loss: sum to 1',
           'loss: keep ratios', 'loss: total'])
plt.grid()
plt.show()

# %%

x, y

# %%

# %%
yy = y.cpu().detach().numpy()
np.sum(yy[:, 0]), np.sum(yy[:, 0] * yy[:, 1]), yy

# %%
