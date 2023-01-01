
# %%
import torch
import torch.nn as nn
import torchvision

import numpy as np

import pandas as pd
import plotly.express as px

# %%
torch.cuda.is_available()

# %%
lr = 1e-2

# %%

sig = nn.Tanh()
sig.cuda()


def mk_pair():
    mlp = torchvision.ops.MLP(2, [4, 3, 2])
    # opt = torch.optim.SGD(mlp.parameters(), lr)
    opt = torch.optim.AdamW(mlp.parameters(), lr)

    mlp.cuda()

    return mlp, opt


mlp, opt = mk_pair()
mlp

# %%
# Target
n = 5000
theta = np.linspace(0, np.pi * 5, n)
r = np.linspace(0.2, 0.8, n)

x = r * np.cos(theta)
y = r * np.sin(theta)

target_data_table = pd.DataFrame()
target_data_table['x'] = x
target_data_table['y'] = y
target_data_table['theta'] = theta
target_data_table['r'] = r
target_data_table['name'] = 'target'

fig = px.scatter(target_data_table, x='x', y='y', color='theta')
fig.show()

target_data_table

# %%
# Source
n = 5000
theta = np.linspace(0, np.pi * 4, n)
r = np.linspace(0.1, 0.9, n)
phi = 0.2

x = r * np.cos(theta + phi)
y = r * np.sin(theta + phi)

src_data_table = pd.DataFrame()
src_data_table['x'] = x
src_data_table['y'] = y
src_data_table['theta'] = theta
src_data_table['r'] = r
src_data_table['name'] = 'src'

fig = px.scatter(src_data_table, x='x', y='y', color='theta')
fig.show()

src_data_table


# %%
data = np.concatenate([
    target_data_table[['x', 'y']].to_numpy(),
    src_data_table[['x', 'y']].to_numpy(),
], axis=1).astype(np.float32)

data.shape

# %%
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# criterion = nn.KLDivLoss()

# %%
lst = []


for j in range(1000):
    np.random.shuffle(data)

    inp = torch.Tensor(data[:100, 2:]).cuda()
    d_target = data[:100, :2]
    target = torch.Tensor(d_target).cuda()

    out = sig(mlp(inp))

    loss = criterion(out, target)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if j % 1 == 0:
        print(j, loss.sum().item())

    lst.append((out.detach().cpu().numpy()[:10]))

    pass

# %%
data_table = []
for j, array in enumerate(lst):
    d = pd.DataFrame(array, columns=['x', 'y'])
    d['name'] = 'node'
    d['j'] = j
    data_table.append(d)

data_table = pd.concat(data_table)
data_table

# %%
_df = data_table.copy()

fig = px.scatter(_df, x='x', y='y', color='j', symbol='name')
fig.show()


# %%
inp = torch.tensor(data[:, 2:]).cuda()
d_target = data[:, :2]
out = sig(mlp(inp))

df1 = pd.DataFrame(out.detach().cpu().numpy(), columns=['x', 'y'])
df1['name'] = 'pred'

df2 = pd.DataFrame(d_target, columns=['x', 'y'])
df2['name'] = 'target'

df = pd.concat([df1, df2])

fig = px.scatter(df, x='x', y='y', color='name')
fig.show()

# %%
df = pd.concat([target_data_table, src_data_table, df1])
fig = px.scatter(df, x='x', y='y', color='name')
fig.show()

# %%
