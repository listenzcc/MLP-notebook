# %%
import torch
import torch.nn as nn
import torchvision

import pandas as pd
import plotly.express as px

# %%
torch.cuda.is_available()

# %%
lr = 1e-3
# %%

sig = nn.Sigmoid()
sig.cuda()


def mk_pair():
    mlp = torchvision.ops.MLP(10, [5, 4, 3])
    # opt = torch.optim.SGD(mlp.parameters(), lr)
    opt = torch.optim.Adam(mlp.parameters(), lr)

    mlp.cuda()

    return mlp, opt


def mk_inp(batch_size=100):
    inp = torch.randn((batch_size, 10))

    inp = inp.cuda()

    return inp


mlp1, opt1 = mk_pair()
mlp2, opt2 = mk_pair()


# %%

# criterion = nn.CrossEntropyLoss()
criterion = nn.KLDivLoss()

# %%
lst = []
for j in range(1000):
    inp1 = mk_inp()
    out1 = sig(mlp1(inp1))

    inp2 = mk_inp()
    out2 = sig(mlp2(inp2))

    loss1 = criterion(out1, out2)
    loss2 = criterion(out2, out1)

    opt1.zero_grad()
    opt2.zero_grad()

    if j % 2 == 0:
        loss1.backward()
    else:
        loss2.backward()

    opt1.step()
    opt2.step()

    if j % 10 == 0:
        print(j, loss1.item(), loss2.item())

    lst.append((out1.detach().cpu().numpy()[:10],
                out2.detach().cpu().numpy()[:10]))

    pass

# %%
data = []
for j, pair in enumerate(lst):
    for name, array in zip(('a', 'b'), pair):
        d = pd.DataFrame(array, columns=['x', 'y', 'z'])
        d['name'] = name
        d['j'] = j
        data.append(d)

data = pd.concat(data)
data

# %%

# %%
_df = data.copy()

_df['size'] = 1

kwargs = dict(
    size='size',
    size_max=5
)
fig = px.scatter_3d(_df, x='x', y='y', z='z',
                    color='j', symbol='name', **kwargs)

for d in fig.data:
    d['marker']['line']['width'] = 0

fig.show()


# %%
