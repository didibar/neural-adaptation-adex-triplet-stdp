import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors

# ----------------------------------------------------------------------
# Global settings and font sizes
plt.rcParams.update({'font.size': 14})

# Font sizes are defined here so all plots can be updated from one place.
fontsize_title = 18        # subplot titles
fontsize_axis_label = 16   # axis labels for t-SNE plots
fontsize_tick = 16         # axis tick labels for heatmaps and t-SNE plots
fontsize_cbar_tick = 16    # colorbar tick labels
fontsize_cbar_label = 14   # colorbar label size, kept for completeness

# ----------------------------------------------------------------------
# Parameters and file paths
alpha = 0.798
beta = 0.727
epsilon = 0.151
seed = 42

dataset_original = {
    "matrix_path": "/home/dilettabartolini/neural-adaptation-adex-triplet-stdp/Results/features_normalized_42/step_0/X_features_normalized.npy",
    "categories": ["wild_animals", "fruit", "flowers", "insects", "birds",
                   "manmade_food", "clothes", "furniture", "instruments", "computer"],
    "name": "DINO FVs"
}

dataset_synthetic = {
    "matrix_path": f"/home/dilettabartolini/neural-adaptation-adex-triplet-stdp/Results/alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}_42/step_1/X_alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}.npy",
    "categories": ["category_1", "category_2", "category_3", "category_4", "category_5",
                   "category_6", "category_7", "category_8", "category_9", "category_10"],
    "name": "SFVs"
}

datasets = [dataset_original, dataset_synthetic]

# Category colors (10 colors)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
rgb_colors = [mcolors.to_rgb(c) for c in colors]

# --- Helper functions for labels and ticks ---

# Create axis labels by showing only the first sample of each category.
def make_labels(categories, samples_per_block=10):
    
    n_categories = len(categories)
    raw_labels = [categories[i // samples_per_block] for i in range(n_categories * samples_per_block)]
    labels = []
    seen = set()
    for lbl in raw_labels:
        if lbl not in seen:
            labels.append(lbl)
            seen.add(lbl)
        else:
            labels.append("")
    return labels


# Create a figure with two heatmaps displayed side by side.
def plot_heatmap_side_by_side(mat1, mat2, labels1, labels2, title1, title2,
                              vmin, vmax, cmap, filename, folder, shared_colorbar=True):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # First heatmap
    im1 = ax1.imshow(mat1, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_title(title1, fontsize=fontsize_title, pad=20)
    positions = [i for i, lbl in enumerate(labels1) if lbl != ""]
    unique_labels = [labels1[i] for i in positions]
    ax1.set_xticks([p for p in positions])
    ax1.set_yticks([p for p in positions])
    ax1.set_xticklabels(unique_labels, rotation=45, ha='right', fontsize=fontsize_tick)
    ax1.set_yticklabels(unique_labels, fontsize=fontsize_tick)

    # Second heatmap
    im2 = ax2.imshow(mat2, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_title(title2, fontsize=fontsize_title, pad=20)
    positions2 = [i for i, lbl in enumerate(labels2) if lbl != ""]
    unique_labels2 = [labels2[i] for i in positions2]
    ax2.set_xticks([p for p in positions2])
    ax2.set_yticks([p for p in positions2])
    ax2.set_xticklabels(unique_labels2, rotation=45, ha='right', fontsize=fontsize_tick)
    ax2.set_yticklabels(unique_labels2, fontsize=fontsize_tick)

    if shared_colorbar:
        cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.05)
        cbar.ax.tick_params(labelsize=fontsize_cbar_tick)
    else:
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.05, pad=0.05)
        cbar1.ax.tick_params(labelsize=fontsize_cbar_tick)
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.05, pad=0.05)
        cbar2.ax.tick_params(labelsize=fontsize_cbar_tick)

    plt.tight_layout()
    plt.savefig(folder / filename, format='pdf')
    plt.close()


# Create a two-panel figure with cosine similarity and t-SNE for one dataset.
def plot_cosine_and_tsne_for_dataset(X, categories, name, samples_per_block, perplexity, folder):

    # Compute cosine similarity.
    cosine_mat = cosine_similarity(X)
    labels = make_labels(categories, samples_per_block)

    # Compute t-SNE embedding.
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    n_blocks = X.shape[0] // samples_per_block
    tsne_labels = np.repeat(np.arange(n_blocks), samples_per_block)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot (a): cosine similarity heatmap with vmin=0.85 and vmax=1.0.
    im = ax1.imshow(cosine_mat, cmap='viridis', vmin=0.85, vmax=1.0, aspect='auto')
    ax1.set_title(f"(a) Cosine Similarity between {name}", fontsize=fontsize_title, pad=20)
    positions = [i for i, lbl in enumerate(labels) if lbl != ""]
    unique_labels = [labels[i] for i in positions]
    ax1.set_xticks([p for p in positions])
    ax1.set_yticks([p for p in positions])
    ax1.set_xticklabels(unique_labels, rotation=45, ha='right', fontsize=fontsize_tick)
    ax1.set_yticklabels(unique_labels, fontsize=fontsize_tick)
    cbar = fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.05, pad=0.05)
    cbar.ax.tick_params(labelsize=fontsize_cbar_tick)

    # Subplot (b): t-SNE embedding.
    cmap = ListedColormap(rgb_colors[:len(categories)])
    norm = BoundaryNorm(np.arange(len(categories)+1)-0.5, cmap.N)
    sc = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tsne_labels, cmap=cmap, norm=norm,
                     alpha=0.7, edgecolor='k', linewidth=0.3)
    ax2.set_title(f"(b) t-SNE embedding of {name}", fontsize=fontsize_title, pad=20)
    ax2.set_xlabel("t-SNE 1", fontsize=fontsize_axis_label)
    ax2.set_ylabel("t-SNE 2", fontsize=fontsize_axis_label)
    ax2.tick_params(labelsize=fontsize_tick)
    ax2.grid(True, linestyle='--', alpha=0.5)
    cbar2 = fig.colorbar(sc, ax=ax2, ticks=np.arange(len(categories)))
    cbar2.ax.set_yticklabels(categories, fontsize=fontsize_cbar_tick)
    cbar2.ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(folder / f"cosine_tsne_{name.replace(' ', '_').lower()}.pdf", format='pdf')
    plt.close()


# Create a figure with two t-SNE plots displayed side by side.
def plot_tsne_side_by_side(X1, X2, categories1, categories2, title1, title2,
                           samples_per_block, perplexity, filename, folder):

    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    X1_tsne = tsne.fit_transform(X1)
    X2_tsne = tsne.fit_transform(X2)

    n_blocks1 = X1.shape[0] // samples_per_block
    n_blocks2 = X2.shape[0] // samples_per_block
    labels1 = np.repeat(np.arange(n_blocks1), samples_per_block)
    labels2 = np.repeat(np.arange(n_blocks2), samples_per_block)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    cmap = ListedColormap(rgb_colors[:len(categories1)])
    norm = BoundaryNorm(np.arange(len(categories1)+1)-0.5, cmap.N)
    sc1 = ax1.scatter(X1_tsne[:, 0], X1_tsne[:, 1], c=labels1, cmap=cmap, norm=norm,
                      alpha=0.7, edgecolor='k', linewidth=0.3)
    ax1.set_title(title1, fontsize=fontsize_title, pad=20)
    ax1.set_xlabel("t-SNE 1", fontsize=fontsize_axis_label)
    ax1.set_ylabel("t-SNE 2", fontsize=fontsize_axis_label)
    ax1.tick_params(labelsize=fontsize_tick)
    ax1.grid(True, linestyle='--', alpha=0.5)

    cmap2 = ListedColormap(rgb_colors[:len(categories2)])
    norm2 = BoundaryNorm(np.arange(len(categories2)+1)-0.5, cmap2.N)
    sc2 = ax2.scatter(X2_tsne[:, 0], X2_tsne[:, 1], c=labels2, cmap=cmap2, norm=norm2,
                      alpha=0.7, edgecolor='k', linewidth=0.3)
    ax2.set_title(title2, fontsize=fontsize_title, pad=20)
    ax2.set_xlabel("t-SNE 1", fontsize=fontsize_axis_label)
    ax2.set_ylabel("t-SNE 2", fontsize=fontsize_axis_label)
    ax2.tick_params(labelsize=fontsize_tick)
    ax2.grid(True, linestyle='--', alpha=0.5)

    cbar1 = fig.colorbar(sc1, ax=ax1, ticks=np.arange(len(categories1)))
    cbar1.ax.set_yticklabels(categories1, fontsize=fontsize_cbar_tick)
    cbar2 = fig.colorbar(sc2, ax=ax2, ticks=np.arange(len(categories2)))
    cbar2.ax.set_yticklabels(categories2, fontsize=fontsize_cbar_tick)
    for cbar in [cbar1, cbar2]:
        cbar.ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(folder / filename, format='pdf')
    plt.close()


# ----------------------------------------------------------------------
# Create the output folder.
output_folder = Path("Input_analysis")
output_folder.mkdir(exist_ok=True)

# Load datasets.
X_orig = np.load(dataset_original["matrix_path"])
X_synth = np.load(dataset_synthetic["matrix_path"])

samples_per_block = 10

# 1. Figure: cosine similarity (0.85-1.0) and t-SNE for the synthetic dataset (SFVs).
plot_cosine_and_tsne_for_dataset(
    X_synth,
    dataset_synthetic["categories"],
    dataset_synthetic["name"],
    samples_per_block,
    perplexity=10,
    folder=output_folder
)

# 2. Figure: side-by-side Euclidean distance heatmaps with separate colorbars.
labels_orig = make_labels(dataset_original["categories"], samples_per_block)
labels_synth = make_labels(dataset_synthetic["categories"], samples_per_block)

euclidean_orig = pairwise_distances(X_orig, metric='euclidean')
euclidean_synth = pairwise_distances(X_synth, metric='euclidean')

plot_heatmap_side_by_side(
    euclidean_orig, euclidean_synth,
    labels_orig, labels_synth,
    f"(a) Euclidean distance between {dataset_original['name']}",
    f"(b) Euclidean distance between {dataset_synthetic['name']}",
    vmin=None, vmax=None, cmap='viridis',
    filename="euclidean_distance_comparison.pdf",
    folder=output_folder,
    shared_colorbar=False   # Use separate colorbars.
)

# 3. Figure: side-by-side t-SNE plots for the original and synthetic datasets with separate colorbars.
plot_tsne_side_by_side(
    X_orig, X_synth,
    dataset_original["categories"], dataset_synthetic["categories"],
    f"(a) t-SNE embedding of {dataset_original['name']}",
    f"(b) t-SNE embedding of {dataset_synthetic['name']}",
    samples_per_block=10, perplexity=10,
    filename="tsne_comparison.pdf",
    folder=output_folder
)
