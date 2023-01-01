'''
File: experiment-1.py

The experiment builds two spiral shape node clouds,
and try to predict one from another.
'''

# %%
import torch
import torch.nn as nn
import torchvision

import numpy as np

import pandas as pd
import plotly.express as px

# %%


def mk_dataframe(x, y, r, theta, name='noname'):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['theta'] = theta
    df['r'] = r
    df['name'] = name
    df['idx'] = range(len(df))
    return df


n = 5000
m = 2000
sample = np.random.choice(range(n), m, replace=False)

# src
theta = np.linspace(0, np.pi * 5, n)
r = np.linspace(0.2, 0.8, n)
x = r * np.cos(theta)
y = r * np.sin(theta)
df1 = mk_dataframe(x, y, r, theta, 'src')
df1['sample'] = 'na'
df1.loc[sample, 'sample'] = 'sample'

# target
theta = np.linspace(0, np.pi * 6, n)
r = np.linspace(0.1, 0.9, n)
x = r * np.cos(theta)
y = r * np.sin(theta)
df2 = mk_dataframe(x, y, r, theta, 'target')
df2['sample'] = 'na'
df2.loc[sample, 'sample'] = 'sample'

# Display
raw_data = pd.concat([df1, df2])

df = raw_data.copy()
df['size'] = 1

size_kwargs = dict(
    size='size',
    size_max=3,
    color_continuous_scale=px.colors.sequential.Turbo
)

fig = px.scatter(df, x='x', y='y', color='name', symbol='sample')
fig.show()

fig = px.scatter_3d(df, x='x', y='y', z='idx', title='spiral-1',
                    color='name', symbol='sample', **size_kwargs)
for d in fig.data:
    d['marker']['line']['width'] = 0
fig.update_layout(legend_orientation="h")
fig.show()

fig = px.scatter_3d(df, x='x', y='y', z='idx', title='spiral-2',
                    color='idx', symbol='sample', **size_kwargs)
for d in fig.data:
    d['marker']['line']['width'] = 0
fig.update_layout(legend_orientation="h")
fig.show()

# %%
train_data = np.concatenate([
    raw_data.query('name=="src"').query(
        'sample=="sample"')[['x', 'y']].to_numpy(),
    raw_data.query('name=="target"').query(
        'sample=="sample"')[['x', 'y']].to_numpy(),
], axis=1)
print('train_data:', train_data.shape)

test_data = np.concatenate([
    raw_data.query('name=="src"').query(
        'sample=="na"')[['x', 'y']].to_numpy(),
    raw_data.query('name=="target"').query(
        'sample=="na"')[['x', 'y']].to_numpy(),
], axis=1)
print('test_data:', test_data.shape)

# %%


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torchvision.ops.MLP(2, [4, 3, 2], activation_layer=nn.LeakyReLU)
        self.mlp2 = torchvision.ops.MLP(2, [3, 2], activation_layer=nn.LeakyReLU)
        self.sig = nn.Tanh()

    def forward(self, x):
        return self.sig(self.mlp2(self.mlp(x)))


net = Net().cuda()

lr = 1e-2
optimizer = torch.optim.AdamW(net.parameters(), lr)
criterion = nn.MSELoss()

net

# %%
test_inp = torch.Tensor(test_data[:, :2]).cuda()
test_trg = torch.Tensor(test_data[:, 2:]).cuda()

for j in range(1000):
    np.random.shuffle(train_data)
    inp = torch.Tensor(train_data[:100, :2]).cuda()
    trg = torch.Tensor(train_data[:100, 2:]).cuda()

    out = net(inp)

    loss = criterion(out, trg)
    loss_test = criterion(net(test_inp), test_trg)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if j % 10 == 0:
        print('Step {:04d}, Loss {:0.4f}, {:0.4f}'.format(
            j, loss.item(), loss_test.item()))

    pass

# %%
output = net(test_inp).detach().cpu().numpy()
df1 = pd.DataFrame(output, columns=['x', 'y'])
n = len(df1)
df1['idx'] = range(n)
df1['name'] = 'pred'

df2 = pd.DataFrame(test_data[:, :2], columns=['x', 'y'])
df2['idx'] = range(n)
df2['name'] = 'truth-dst'

df3 = pd.DataFrame(test_data[:, 2:], columns=['x', 'y'])
df3['idx'] = range(n)
df3['name'] = 'truth-src'

df4 = pd.DataFrame(output - test_data[:, :2], columns=['x', 'y'])
df4['idx'] = range(n)
df4['name'] = 'pred - dst'

df = pd.concat([df1, df2, df3, df4])
df['size'] = 1

fig = px.scatter(df, x='x', y='y', color='name', opacity=0.5)
fig.show()

fig = px.scatter_3d(df, x='x', y='y', z='idx',
                    title='spiral-1', color='name', **size_kwargs)
for d in fig.data:
    d['marker']['line']['width'] = 0
fig.update_layout(legend_orientation="h")
fig.show()

fig = px.scatter_3d(df, x='x', y='y', z='idx', symbol='name',
                    title='spiral-2', color='idx', **size_kwargs)
for d in fig.data:
    d['marker']['line']['width'] = 0
fig.update_layout(legend_orientation="h")
fig.show()

# %%
