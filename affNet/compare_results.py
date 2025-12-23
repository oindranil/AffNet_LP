"""
program:
    retrieves homophily (aff, edge & node) scores of different datasets
    compares and creates reports
"""

# setup environment
env = 'windows' 
if env=='windows':
    root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/codebase/affNet/'
    data_folder = "D:/Indranil/ML2/Datasets/"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
###
### main program ###
###

# datasets
datasets = ['Texas', 'Wisconsin', 'Cora', 'CiteSeer', 'Photo', 'Squirrel', 'Chameleon']

results_df = pd.read_csv(root+'results/results.csv')[['dataset', 'aff_h']]
homophily_df = pd.read_csv(root+'params/homophily.csv')[['dataset', 'node_h', 'edge_h']]
df = pd.merge(results_df, homophily_df, on='dataset', how='inner')
df.sort_values(by='edge_h', axis=0, ascending=False, inplace=True)
df = df[df['dataset'].isin(datasets)]

X_axis = np.arange(len(df))
if len(df)>1:
    # plot results
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]  

    plt.figure(figsize=(9, 5), dpi=600)
    plt.bar(X_axis + offsets[0], df['node_h'].values, bar_width, label='Node-homophily', hatch='\\\\')
    plt.bar(X_axis + offsets[1], df['edge_h'].values, bar_width, label='Edge-homophily', hatch='.')
    plt.bar(X_axis + offsets[2], df['aff_h'].values, bar_width, label='Affinity-homophily', hatch='//')

    plt.xticks(X_axis-0.25, df['dataset'].values, rotation=45, fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylabel("Homophily", fontsize=16)
    plt.ylim(0, 1)
    xmin, xmax = plt.xlim()
    plt.hlines(0.5, xmin, xmax, 'b', 'dotted')
    plt.text(3.0, 0.55, '$\mathit{middle\ line\: (0.5)}$', fontsize=14, color='k')
    plt.text(0.15, 0.90, 'Homophily', fontsize=14, color='b')
    plt.text(4.5, 0.4, 'Heterophily', fontsize=14, color='b')
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{root}results/comparison.pdf')
    plt.show()
