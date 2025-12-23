"""
program: plot heads ablation results
    
"""

# setup environment
env = 'windows' 
if env=='windows':
	root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/codebase/affNet/'
	data_folder = "D:/Indranil/ML2/Datasets/"

import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv(root+"results/heads.csv")
results_df.sort_values(by = ['dataset', 'heads'], inplace=True)
datasets = results_df['dataset'].unique()
datasets = ['Cora', 'Texas', 'Chameleon', 'Wisconsin']
colors = ['cyan', 'yellow', 'magenta', 'blue', 'red', 'green', 'brown', 'pink', 'violet']

x = [1, 2, 4, 8, 16]
i = 0
plt.figure(figsize=(8,6), dpi=600)
for dataset in datasets:
    results = results_df[results_df['dataset']==dataset][['heads', 'metric_mean', 'metric_std']]
    plt.subplot(2,2,i+1)
    x, y, error = results['heads'].values,results['metric_mean'].values, results['metric_std'].values
    plt.plot(x, y, marker='x', label=dataset, color='k')
    plt.fill_between(x, y-error, y+error, color=colors[i])
    plt.title(dataset, fontsize=18)
    plt.ylim(0.7,1)
    plt.xscale('log', base=2)
    plt.xticks(ticks=x, labels=[str(val) for val in x], fontsize=16)
    plt.yticks([0.7, 0.8, 0.9, 1.0], fontsize=16)
    if i in [2,3]:
        plt.xlabel("Heads", fontsize=16)
    else:
        plt.xticks([])
    if i in [0,2]:
        plt.ylabel("AUC", fontsize=16)
    else:
        plt.yticks([])
    i = i+1
plt.tight_layout()
plt.savefig(f"{root}results/heads.pdf")
plt.show()

