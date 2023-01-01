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


def mk_pair():
    mlp = torchvision.ops.MLP(4, [4, 3, 2])
    # opt = torch.optim.SGD(mlp.parameters(), lr)
    opt = torch.optim.AdamW(mlp.parameters(), lr)

    mlp.cuda()

    return mlp, opt


def mk_inp(batch_size=100):
    inp = torch.randn((batch_size, 5))

    inp = inp.cuda()

    return inp


mlp, opt = mk_pair()
mlp

# %%
# Ground truth
n = 5000
theta = np.linspace(0, np.pi * 5, n)
r = np.linspace(0.2, 0.8, n)

x = r * np.cos(theta)
y = r * np.sin(theta)

raw_data_table = pd.DataFrame()
raw_data_table['x'] = x
raw_data_table['y'] = y
raw_data_table['theta'] = theta
raw_data_table['r'] = r

fig = px.scatter(raw_data_table, x='x', y='y', color='theta')
fig.show()

raw_data_table

# %%
data = np.random.randn(len(raw_data_table), 2+8).astype(np.float32)
data[:, :2] = np.array(raw_data_table[['x', 'y']])
np.random.shuffle(data)
data.shape

# %%


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        embed_dim = 8
        num_heads = 1

        self.mk_key = torchvision.ops.MLP(8, [5, 8])
        self.mk_value = torchvision.ops.MLP(8, [5, 8])
        self.mk_query = torchvision.ops.MLP(8, [8, 8])
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.tanh = nn.Tanh()
        self.mlp = torchvision.ops.MLP(8, [4, 2])

    def forward(self, x):
        query = self.mk_query(x)
        key = self.mk_key(x)
        value = self.mk_value(x)
        attn_output, attn_output_weights = self.multihead_attn(
            query, key, value)

        middle = attn_output

        output = self.tanh(self.mlp(middle))

        return output


net = MyNet()
opt = torch.optim.Adam(net.parameters(), lr)
net.cuda()

inp = torch.randn((10, 8))
output = net(inp.cuda())
output.shape


# %%
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# criterion = nn.KLDivLoss()

# %%
lst = []


for j in range(1000):
    np.random.shuffle(data)

    inp = torch.Tensor(data[:1000, 2:]).cuda()
    d_target = data[:1000, :2]
    target = torch.Tensor(d_target).cuda()

    out = net(inp)

    loss = criterion(out, target)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if j % 1 == 0:
        print(j, loss.item())

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
out = net(inp)

df1 = pd.DataFrame(out.detach().cpu().numpy(), columns=['x', 'y'])
df1['name'] = 'pred'

df2 = pd.DataFrame(d_target, columns=['x', 'y'])
df2['name'] = 'raw'

df = pd.concat([df1, df2])

fig = px.scatter(df, x='x', y='y', color='name')
fig.show()

# %%

# query = torch.randn((10, embed_dim))
# key = torch.randn((5, embed_dim))
# value = torch.randn((5, embed_dim))

# attn_output, attn_output_weights = multihead_attn(query, key, value)

# attn_output.shape, attn_output_weights.shape

# %%

# %%
