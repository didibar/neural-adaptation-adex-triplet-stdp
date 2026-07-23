import numpy as np
from brian2 import *
from pathlib import Path
import shutil

import step_0_DINO_FVs as st0            # Input extraction using DINO
import step_1_SFVs as st1                # Input syntetic feature vector generation
import step_2_learning as st2            # Network learning module
import step_seq_generator as stseq       # Sequence generation module
import step_3_testing as st3             # AdEx neuron simulation
import step_4_mean_firing_rate as st4    # Result analysis for firing-rate dynamics
import step_5_onset_latency as st5       # Onset latency analysis
import step_6_population_latency as st6  # Population latency analysis

import warnings
warnings.filterwarnings("ignore")


# Prepare a clean Brian2 standalone simulation environment.
# The function recreates the build directory, resets Brian2, and activates the C++ standalone backend.
def prepare_simulation():
    output_dir = Path('standalone_code')

    # Remove any previous standalone build to avoid stale compiled files.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create the standalone build directory before activating the device.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset the Brian2 state and configure the C++ standalone backend.
    start_scope()
    set_device('cpp_standalone', directory=str(output_dir), build_on_run=False)
    device.reinit()
    device.activate()


# Initialize the Brian2 scope before running the full pipeline.
start_scope()

# Define the parameters used to generate the structured feature vectors.
alpha =  0.798
beta = 0.727     
epsilon = 0.151
seed = 42

# Create the input-pattern suffix used to name result folders and saved matrices.
file_suffix = f"alpha{alpha}_beta{beta}_eps{epsilon}_seed{seed}"
#file_suffix = "features_normalized"

# Create the directory structure used to save all pipeline outputs.
results_dir = Path("Results")
final_dir = results_dir / f"{file_suffix}_{seed}"
results_dir.mkdir(exist_ok=True)
final_dir.mkdir(exist_ok=True)

step_dirs = []
for i in range(7):
    step_dir = final_dir / f"step_{i}"
    step_dir.mkdir(exist_ok=True)
    step_dirs.append(step_dir)

# Step 1: create or load the input feature matrix.
if file_suffix == "features_normalized":
    image_folder = Path("/home/dilettabartolini/neural-adaptation-adex-triplet-stdp/dataset")
    x = st0.prepare_input_matrix(image_folder, step_dirs[0])
    n_neurons = x.shape[1]
    print(f"Input matrix shape: {x.shape}")
    print(f"Number of neurons determined from input matrix: {n_neurons}")

else:
    n_neurons = 768
    x = st1.generate_block_structured_input(alpha, beta, epsilon, n_neurons, seed)
    np.save(step_dirs[1] / f"X_{file_suffix}.npy", x)


# Step 2: train the recurrent AdEx network and save its learned connectivity.
base_path = Path(f"/home/dilettabartolini/neural-adaptation-adex-triplet-stdp/Results/{file_suffix}_{seed}")

n_motif = 10

set_device("runtime")

st2.learn_and_save_network(file_suffix, n_motif, seed, base_path, step_dirs[2])


# Step 3: generate test sequences and run the trained network on each sequence.
sequence_length = 5

stseq.generate_sequences(n_motif, sequence_length, seed, step_dirs[3])

test_sequences_csv = step_dirs[3] / "combined_dataset_matrix.csv"

prepare_simulation()

total_time = st3.AdEx_test(file_suffix, seed, base_path, step_dirs[3], test_sequences_csv)

dt = 0.1             # Simulation time step, in ms.
initial_noise = 100  # Initial random-noise period in ms.

# Compute the number of time points associated with each image presentation.
n_time_each = int(((1600 - initial_noise) / dt) / sequence_length) # total_time

activations_dir = step_dirs[3] / "Activations_test" / file_suffix

matrices_dict = st3.load_matrices_with_metadata(activations_dir)

output_dir_st3 = step_dirs[3] / "Processed_matrices"

output_dir_st3.mkdir(parents=True, exist_ok=True)

results_st3 = st3.process_matrices(sequence_length, matrices_dict, n_neurons, n_time_each, output_dir_st3)

# Load the processed Primed and Control activation matrices.
p_matrix = np.load(output_dir_st3 / "P_extended.npy")  # results['P_extended']
c_matrix = np.load(output_dir_st3 / "C_extended.npy")  # results['C_extended']

# Load the stimulus indices associated with each processed activation matrix.
p_index = np.load(output_dir_st3 / "P_extended_indices.npy")  # results['P_indices']
c_index = np.load(output_dir_st3 / "C_extended_indices.npy")  # results['C_indices']

n_pres = p_index.shape[0]  # Number of presentations.

# Step 4: analyze population-level firing-rate dynamics.
results_st4 = st4.population_level_firing_rate_dynamics(
    p_matrix,
    c_matrix,
    step_dirs[4],
    n_pres,
    dt,
    seed,
    n_permutations=100000,
    alpha=0.05,
    min_cluster_size=5,
    t_threshold=None
)

# Store significant-duration summaries.
p_g_c = results_st4['total_primed_gt_control']
c_g_p = results_st4['total_primed_lt_control']

# Step 5: analyze onset latency across the entire network.
results_onset = st5.onset_latency(
    p_matrix,
    c_matrix,
    n_pres=n_pres,
    n_neurons_total=n_neurons,
    n_windows=sequence_length,
    dt=dt,
    output_dir=step_dirs[5],
    min_onset=0,
    max_onset=300,
    n_permutations=100000,
    alpha=0.05,
    random_seed=seed,
    correction_method='bonferroni',
    min_active_presentations=3,
    baseline_window=100
)

# Step 6: analyze activation timing for the most responsive neurons in the network.
threshold1 = 10  # Activation threshold for the first analysis.
threshold2 = 15  # Activation threshold for the second analysis.
results = st6.population_latency(p_matrix, c_matrix, dt, threshold1, threshold2, sequence_length, n_pres, step_dirs[6])
