# %%
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import plotly.express as px

# %%

select_num = 30
trial_repeats = 50

# %%


def simulation(n, m=select_num):
    ''' Simulate the random selection of m elements from n elements '''
    total = np.array(range(n))
    select = sorted(np.random.choice(total, m, replace=False))
    return select


# %%


data = []

for n in tqdm(range(100, 500), 'Making dataset'):
    for _ in range(trial_repeats):
        select = simulation(n)
        data.append((select_num, n, select))

simulation_table = pd.DataFrame(data, columns=['m', 'n', 'select'])
simulation_table


# %%
n = 277

target = simulation(n)


def _diff(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


simulation_table['diff'] = simulation_table['select'].map(
    lambda a: _diff(a, target))

fig = px.scatter(simulation_table, y='diff', color='n',
                 title='Simulation: {}'.format(n))
fig.show()

mean_table = simulation_table.groupby('n').mean()
mean_table['n'] = mean_table.index

fig = px.scatter(mean_table, y='diff', color='n',
                 title='Simulation(mean): {}'.format(n))
fig.show()

# %%
