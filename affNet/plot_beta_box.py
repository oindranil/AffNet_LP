import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_beta_spread(root, filepath, datasets):
    
    # Read CSV
    df = pd.read_csv(filepath)

    # Keep only relevant columns
    df = df[['dataset', 'run_no', 'beta']]
    df = df[df['dataset'].isin(datasets)]

    # Prepare data for boxplot
    beta_data = []
    for dataset in datasets:
        # Filter for current dataset
        beta_values = df[df['dataset'] == dataset].sort_values(by='run_no')['beta']
        
        beta_data.append(beta_values.values)

    # Plotting
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'wheat', 'plum', 'khaki']
    color_cycle = itertools.cycle(colors)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    bp = ax.boxplot(beta_data, labels=datasets, showfliers=False, patch_artist=True)
    # Set x positions for ticks slightly left-shifted
    n = len(datasets)
    positions = np.arange(1, n + 1)  # boxplot uses 1-based positions
    offset = -0.4  # tune this if needed

    ax.set_xticks(positions + offset)
    ax.set_xticklabels(datasets, rotation=45, fontsize=20)
    for patch in bp['boxes']:
        patch.set_facecolor(next(color_cycle))
    #plt.title("Spread of Beta Values Across Runs for Individual Datasets")
    plt.ylabel("Beta Value", fontsize=20)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{root}results/beta_box.pdf')
    plt.show()

root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/codebase/affNet/'
filepath = root+'results/beta_values.csv'
datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin', 'Squirrel', 'Chameleon', 
            'Photo','ogbl-ppa', 'ogbl-collab', 'ogbl-citation2']
plot_beta_spread(root, filepath, datasets)

