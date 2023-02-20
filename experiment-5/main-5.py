
'''
File: main-5.py

The experiment trys the MPL to learn phase from complex number.
'''

# %%
import torch
import torch.nn as nn
import torchvision

import numpy as np

import pandas as pd
import plotly.express as px

# %%
n, m = 10000, 2000

real = np.random.randn(n)
imag = np.random.randn(n)
phase = np.arctan2(imag, real)

data_table = pd.DataFrame(real, columns=['real'])
data_table['imag'] = imag
data_table['phase'] = phase / np.pi
data_table['set'] = 'train'
data_table.loc[:m, 'set'] = 'test'
data_table

# %%
print(np.max(data_table['phase']), np.min(data_table['phase']))

kwargs = {'width': 600, 'height': 600}

fig = px.scatter_3d(data_table, x='real', y='imag',
                    z='phase', color='phase', **kwargs)
fig.data[0]['marker']['size'] = 3
fig.show()

fig = px.scatter(data_table, x='real', y='imag', color='phase', **kwargs)
fig.show()

# %%


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torchvision.ops.MLP(
            2, [4, 8, 4, 1], activation_layer=nn.LeakyReLU)
        self.sig = nn.Tanh()

    def forward(self, x):
        return self.sig(self.mlp(x))


net = Net().cuda()

lr = 1e-2
optimizer = torch.optim.AdamW(net.parameters(), lr)
criterion = nn.MSELoss()

net
# %%
train_data = np.array(data_table.query('set == "train"')
                      [['real', 'imag', 'phase']])
test_data = np.array(data_table.query('set == "test"')
                     [['real', 'imag', 'phase']])
print(train_data.shape, test_data.shape)

test_inp = torch.Tensor(test_data[:, :2]).cuda()
test_trg = torch.Tensor(test_data[:, 2:]).cuda()

training_loss = []
for j in range(2000):
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
    training_loss.append(loss.item())

    pass

fig = px.scatter(training_loss, title='Training Loss')
fig.show()

# %%

# %%

df = data_table.query('set == "test"').copy()
output = net(test_inp).detach().cpu().numpy()
df['predPhase'] = output
df['diff'] = df['predPhase'] - df['phase']
df['diff'] = df['diff'].map(lambda e: e % (np.pi * 2))
df['diff'] = df['diff'].map(lambda e: np.min((e, np.pi * 2 - e)))
df['diffRatio'] = np.abs(df['diff']) / np.abs(df['phase'])
df

# %%
fig = px.scatter_3d(df,
                    x='real', y='imag', z='phase', color='phase', title='phase', **kwargs)
fig.data[0]['marker']['size'] = 3
fig.show()

fig = px.scatter_3d(df,
                    x='real', y='imag', z='predPhase', color='diff', title='predPhase',
                    **kwargs,
                    color_continuous_scale='turbo')
fig.data[0]['marker']['size'] = 3
fig.show()

fig = px.scatter_3d(df,
                    x='real', y='imag', z='diff', color='diff', title='diffPhase',
                    **kwargs,
                    color_continuous_scale='turbo')
fig.data[0]['marker']['size'] = 3
fig.show()

fig = px.histogram(df, x='diff', **kwargs)
fig.show()

fig = px.scatter(df, y='diff', x='phase', color='diff', **kwargs)
fig.show()

# %%
# px.colors.sequential.swatches()
# %%
