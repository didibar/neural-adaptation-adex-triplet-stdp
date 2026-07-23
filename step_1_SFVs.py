import numpy as np
from pathlib import Path

# Dataset layout parameters.
n_blocks = 10
n_macros = 2
subcats_per_macro = 5
samples_per_subcat = 10

# Generate a synthetic input matrix with a block-wise similarity structure.
def generate_block_structured_input(intra_similarity, inter_dissimilarity, epsilon, n_neurons, seed):
    if seed is not None:
        np.random.seed(seed)

    x_obs_path = Path("/home/dilettabartolini/neural-adaptation-adex-triplet-stdp/Results/features_normalized_42/step_0/X_features_normalized.npy")

    # Use the observed feature dimensionality when the reference matrix is available.
    if x_obs_path.exists():
        x_obs = np.load(x_obs_path)
        n_features = x_obs.shape[1]
    else:
        n_features = n_neurons

    total_samples = n_blocks * samples_per_subcat
    input_matrix = np.zeros((total_samples, n_features))

    # Create block centers with controlled dissimilarity from a shared global base.
    global_base = np.random.rand(n_features)
    inter_range = inter_dissimilarity * 2
    block_centers = np.array([
        global_base + np.random.uniform(-inter_range, inter_range, n_features)
        for _ in range(n_blocks)
    ])

    # Generate samples around each block center using intra-cluster noise.
    sample_idx = 0
    for block_idx in range(n_blocks):
        center = block_centers[block_idx]
        intra_noise = np.sqrt(1.0 - intra_similarity)  # Standard deviation for intra-cluster noise.

        for _ in range(samples_per_subcat):
            noise = np.random.normal(0, intra_noise, size=n_features)
            sample = center + noise
            input_matrix[sample_idx] = sample
            sample_idx += 1

    # Reduce dissimilarity within each half of the matrix using epsilon.
    if epsilon > 0:
        half = total_samples // 2
        for half_matrix in [input_matrix[:half], input_matrix[half:]]:
            # Compute the centroid of the current half.
            centroid = np.mean(half_matrix, axis=0)

            # Move each sample towards the centroid by an epsilon-scaled amount.
            for i in range(len(half_matrix)):
                half_matrix[i] = half_matrix[i] + epsilon * (centroid - half_matrix[i])

    # Normalize values to the [0, 1] range.
    input_matrix -= input_matrix.min()
    input_matrix /= input_matrix.max()

    return input_matrix
