
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


# %%
length = 100

[a, b] = np.meshgrid(range(length), range(length))
a = a.ravel() / length
b = b.ravel() / length
a = a[::int(length/100)]
b = b[::int(length/100)]

# z = np.array([0+0j for a, b in zip(a, b)])
# z.real = a
# z.imag = b


# def _rnd():
#     return np.random.random()


# y_total = np.sin(np.abs(
#     # ((z - (_rnd() + _rnd() * 1j)) ** 2) *
#     # ((z ** 2 - (_rnd() + _rnd() * 1j)) ** (1)) *
#     ((z ** 1 - (_rnd() + _rnd() * 1j)) ** (-1)) *
#     ((z ** 1 - (_rnd() + _rnd() * 1j)) ** (-1)) *
#     ((z ** 1 - (_rnd() + _rnd() * 1j)) ** (-1)) *
#     1
# ))

# example-1
# y_total = np.cos((a + b) * 15)

# example-2
# y_total = np.cos((a**0.5 + b**2) * 15)

# example-3
y_total = np.cos((a**0.5 - b**2) * 15)

y_total = y_total[:, np.newaxis]

X_total = np.concatenate([e[:, np.newaxis] for e in [a, b]], axis=1)

print(X_total.shape, y_total.shape)


# %%
num_train = 800
num_valid = 200

X = X_total.copy()
y = y_total.copy()


def _type(x):
    return any([
        (x[0] < 0.2),
        (x[0] > 0.4 and x[0] < 0.6),
        (x[0] > 0.8),
    ])


select = [(j, _type(e)) for j, e in enumerate(X)]

idx_train = [e[0] for e in select if e[1]]
np.random.shuffle(idx_train)
X_train = X[idx_train[:num_train]]
y_train = y[idx_train[:num_train]]

idx_valid = [e[0] for e in select if not e[1]]
np.random.shuffle(idx_valid)
X_valid = X[idx_valid[:num_valid]]
y_valid = y[idx_valid[:num_valid]]

print([e.shape for e in [X_train, y_train, X_valid, y_valid]])

# df1 = pd.DataFrame(X_train, columns=['x', 'y'])
# df1['z'] = y_train
# df1['type'] = 'train'

# df2 = pd.DataFrame(X_valid, columns=['x', 'y'])
# df2['z'] = y_valid
# df2['type'] = 'valid'

# df = pd.concat([df1, df2])
# fig = px.scatter(df, x='x', y='y', color='z', symbol='type')
# fig.show()

# %%
idx_total = list(range(len(y_total)))
np.random.shuffle(idx_total)
X_total = X_total[idx_total[:10000]]
y_total = y_total[idx_total[:10000]]

print(X_total.shape, y_total.shape)

# df = pd.DataFrame(X_total, columns=['x', 'y'])
# df['z'] = y_total
# df['size'] = 1
# fig = px.scatter(df, x='x', y='y', color='z', size='size', size_max=5)
# fig.data[0]['marker']['line']['width'] = 0
# fig.show()

# %% ------------------------------------------
#
#


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torchvision.ops.MLP(
            num_features, [4, 10, 4, 1], activation_layer=nn.LeakyReLU)
        self.act = nn.Tanh()

    def forward(self, x):
        output = self.mlp(x)
        output = self.act(output)
        return output


net = Net().cuda()

# lr = 1e-2
# optimizer = torch.optim.AdamW(net.parameters(), lr)
optimizer = torch.optim.AdamW(net.parameters())
criterion = nn.MSELoss()
net

# %%
training_loss = []
validation_loss = []

X1 = torch.Tensor(X_valid).cuda()
y1 = torch.Tensor(y_valid).cuda()

k = 100
idx_train = list(range(len(y_train)))

for j in tqdm(range(loops), 'Training'):
    np.random.shuffle(idx_train)

    X = torch.Tensor(X_train[idx_train[:k]]).cuda()
    y = torch.Tensor(y_train[idx_train[:k]]).cuda()

    loss = criterion(net(X), y)
    l = loss.item()

    loss_valid = criterion(net(X1), y1)
    l_valid = loss_valid.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    training_loss.append(l)
    validation_loss.append(l_valid)

    if j % 1000 == 0:
        print(j, l, l_valid)

# df1 = pd.DataFrame(training_loss, columns=['loss'])
# df1['type'] = 'training'

# df2 = pd.DataFrame(validation_loss, columns=['loss'])
# df2['type'] = 'validation'

# df_loss = pd.concat([df1, df2])

# fig = px.line(df_loss, y='loss', color='type')
# fig.show()

# %%
X_total_cuda = torch.Tensor(X_total).cuda()
y_total_pred = net(X_total_cuda).detach().cpu().numpy().ravel()
y_total_pred.shape

# df = pd.DataFrame(X_total, columns=['x', 'y'])
# df['z'] = y_total_pred
# df['size'] = 1
# fig = px.scatter(df, x='x', y='y', color='z', size='size', size_max=5)
# fig.data[0]['marker']['line']['width'] = 0
# fig.show()

# %%
jobs = [
    dict(
        title='Ground truth',
        axis=(0, 0),
        kwargs=dict(
            x=X_total[:, 0],
            y=X_total[:, 1],
            c=y_total
        )),
    dict(
        title='Pred',
        axis=(0, 1),
        kwargs=dict(
            x=X_total[:, 0],
            y=X_total[:, 1],
            c=y_total_pred
        )),
    dict(
        title='Training',
        axis=(1, 0),
        kwargs=dict(
            x=X_train[:, 0],
            y=X_train[:, 1],
            c=y_train
        )),
    dict(
        title='Validation',
        axis=(1, 1),
        kwargs=dict(
            x=X_valid[:, 0],
            y=X_valid[:, 1],
            c=y_valid
        )),
    dict(
        title='Loss',
        axis=(2, 0),
        kwargs=None),
    dict(
        title='Diff',
        axis=(2, 1),
        kwargs=dict(
            x=X_total[:, 0],
            y=X_total[:, 1],
            c=y_total_pred.ravel() - y_total.ravel()
        )),
]

plt.style.use(['ggplot'])
fig, axes = plt.subplots(3, 2, figsize=(12, 12), num='seaborn')

for job in jobs:
    ax = axes[job['axis']]
    ax.set_title(job['title'])
    if job['kwargs']:
        im = ax.scatter(vmin=-1, vmax=1,
                        cmap=plt.get_cmap('viridis'), **job['kwargs'])
        fig.colorbar(im, ax=ax)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))

ax = axes[(2, 0)]
ax.plot(training_loss, label='training_loss')
ax.plot(validation_loss, label='validation_loss')
ax.legend()
ax.grid(True)

# fig.colorbar(im)

fig.tight_layout()

# %%
# Switch between immediate show and savefig

if DEBUG:
    fig.show()
else:
    filename = '{}-{}.png'.format(time.time(), np.random.random())
    fig.savefig(filename)
    print('Done with', filename)


# %%
