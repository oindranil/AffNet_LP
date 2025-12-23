import matplotlib.pyplot as plt
import numpy as np

root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/codebase/affNet/'
datasets = ['Cora', 'CiteSeer', 'Texas', 'Wisconsin']

# Define the formats
labels = [
    'Single headed, no thresholding',
    'Multi-headed, no thresholding',
    'Single headed, thresholding',
    'Multi-headed, thresholding']
#linestyles = ['dotted', 'dashed','solid', 'solid' ]
linestyles = ['--',':', '-.', '-']
linewidths = [2, 2, 2, 2]
markers = ['s', 'D', 'o', '^']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()  # Flatten the 2D array of axes for easier iteration

for j in range(4):
    dataset_name = datasets[j]
    hist_fname = f"{root}/results/hist_{dataset_name}.csv"
    hist_data = np.loadtxt(hist_fname, delimiter=",")
    n_points = hist_data.shape[1]
    x = np.arange(n_points) + 1
    epochs = max(x)
    for i in range(4):
        axes[j].plot(x[::200], hist_data[i][::200], label=labels[i], 
                linestyle=linestyles[i], linewidth=linewidths[i])
        axes[j].set_xlabel("epochs", fontsize=24)
        axes[j].set_ylabel("loss", fontsize=24)
        axes[j].set_xticks([])
        axes[j].set_yticks([0, 0.2, 0.4], [0, 0.2, 0.4], fontsize=20)
    axes[j].set_title(dataset_name, fontsize=24)

# Create a common legend at the bottom
fig.legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=24, frameon=False)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(hspace=0.6, bottom=0.18)  # Increase the vertical space
plt.savefig(f"{root}results/learning_curve.pdf", bbox_inches='tight')
plt.show()
