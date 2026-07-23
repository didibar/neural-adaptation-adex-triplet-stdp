import csv

import numpy as np
from brian2 import *
from pathlib import Path
import pickle
import shutil
from warnings import warn
import pandas as pd
from collections import defaultdict
from scipy import stats
import pandas as pd

# Load test sequences from a CSV file.
def load_test_sequences_from_csv(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check that all required columns are available.
    required_cols = ['Sequence_ID', 'Pos_1', 'Pos_2', 'Pos_3', 'Pos_4', 'Pos_5']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column missing: {col}")

    # Create the list of test sequences.
    sequences = []
    for idx, row in df.iterrows():
        seq_name = row['Sequence_ID']
        seq_indices = [
            int(row['Pos_1']),
            int(row['Pos_2']),
            int(row['Pos_3']),
            int(row['Pos_4']),
            int(row['Pos_5'])
        ]

        sequences.append({
            'name': seq_name,
            'indices': seq_indices
        })

    return sequences


# Test a single sequence and save the network activity.
def test_single_sequence(seq_info, feature_matrix, network_state):

    seq_name = seq_info['name']
    seq = seq_info['indices']

    print(f"\n  Testing sequence: {seq_name}")
    print(f"    Indices: {seq}")

    # Parameters
    n_neurons = feature_matrix.shape[1]  # Number of neurons in the network
    n_excitatory = int(3/4 * n_neurons)  # Number of excitatory neurons
    n_inhibitory = int(1/4 * n_neurons)  # Number of inhibitory neurons

    # Neuron parameters
    c_m = 281 * pF
    g_l = 30 * nS
    tau_m = c_m / g_l
    e_l = -70.6 * mV
    v_t = -50.4 * mV
    delta_t = 2 * mV
    v_cut = v_t + 5 * delta_t

    tau_e = 5 * ms
    tau_i = 10 * ms
    e_e = 0 * mV
    e_i = -80 * mV

    # Configuration
    repeats = len(seq)  # Number of features per sequence = 5

    # Temporal parameters
    dt = defaultclock.dt         # Time step (100 us = 0.1 ms)
    stim_duration = 300 * ms     # Duration of the stimulus

    pre_stim_noise_duration = 100 * ms  # Duration of random noise before the first stimulus
    noise_intensity = 0.98              # Intensity of random noise

    total_time = pre_stim_noise_duration + stim_duration * repeats
    total_time_simulated = int(round(float(total_time / ms)))
    time_points = np.arange(0, total_time, dt)

    if len(time_points) % repeats != 0:
        warn(f"Time window not perfectly divisible by {repeats}")

    # Device setup
    device.reinit()
    device.activate()
    set_device('cpp_standalone', build_on_run=False)

    # Define the stimulus input current.
    i_values_test = np.zeros((n_neurons, len(time_points)))
    print(f" i_values_test shape: {i_values_test.shape}")

    noise_end = int(pre_stim_noise_duration / dt)
    time_pre = np.linspace(0, 1, noise_end)
    freq = 1
    phases = np.random.rand(n_neurons, 1) * 2 * np.pi
    i_values_test[:, :noise_end] = np.clip(noise_intensity * np.sin(2 * np.pi * freq * time_pre + phases), 0, 1)

    # Apply each image in the sequence.
    for k in range(repeats):
        image_idx = seq[k]
        stim_start = noise_end + k * int(stim_duration / dt)
        stim_end = stim_start + int(stim_duration / dt)
        i_values_test[:, stim_start:stim_end] = feature_matrix[image_idx, :][:, np.newaxis]

    i_recorded = TimedArray(i_values_test.T * nA, dt=dt)

    # Neuronal dynamics
    adex_eqs = '''
        dvm/dt = (g_l*(e_l - vm) + g_l*delta_t*exp((vm - v_t)/delta_t) + i_inj - wad + g_e*(e_e-vm) + g_i*(e_i-vm)) / c_m : volt

        dwad/dt = (a*(vm - e_l) - wad)/tauw : amp

        dg_e/dt = -g_e/tau_e : siemens        # Excitatory synaptic conductance
        dg_i/dt = -g_i/tau_i : siemens        # Inhibitory synaptic conductance

        i_inj = i_recorded(t, i) : amp
        tauw : second
        a : siemens
        b : amp
        vr : volt
    '''

    # Create excitatory and inhibitory neuron groups.
    ge = NeuronGroup(n_excitatory, adex_eqs, threshold = 'vm > v_cut', reset="vm=vr; wad+=b", method='euler')
    gi = NeuronGroup(n_inhibitory, adex_eqs, threshold = 'vm > v_cut', reset="vm=vr; wad+=b", method='euler')

    # Initialize neuron parameters.
    ge.vm = e_l
    gi.vm = e_l

    ge.g_e = 0 * nS
    ge.g_i = 0 * nS
    gi.g_e = 0 * nS
    gi.g_i = 0 * nS
    gi.wad = 0 * nA
    ge.wad = 0 * nA

    # Assign parameters for regular-spiking excitatory neurons.
    ge.tauw = 144 * ms
    ge.a = 4 * nS
    ge.b = 0.0805 * nA
    ge.vr = -70.6 * mV
    # Assign parameters for fast-spiking inhibitory neurons.
    gi.tauw = 300 * ms
    gi.a = 2 * c_m / (144 * ms)
    gi.b = 0 * nA
    gi.vr = -70.6 * mV

    # Create synaptic connections.
    c_ee = Synapses(ge, ge, model='w_ : siemens', on_pre='g_e_post += w_')
    c_ei = Synapses(ge, gi, model='w_ : siemens', on_pre='g_e_post += w_')
    c_ie = Synapses(gi, ge, model='w_ : siemens', on_pre='g_i_post += w_')
    c_ii = Synapses(gi, gi, model='w_ : siemens', on_pre='g_i_post += w_')

    # Load saved connectivity and weights from the learning phase.
    c_ee.connect(i=network_state["C_ee_i"], j=network_state["C_ee_j"])
    c_ei.connect(i=network_state["C_ei_i"], j=network_state["C_ei_j"])
    c_ie.connect(i=network_state["C_ie_i"], j=network_state["C_ie_j"])
    c_ii.connect(i=network_state["C_ii_i"], j=network_state["C_ii_j"])

    # Set synaptic weights.
    c_ee.w_ = network_state["W_ee"] * nS
    c_ei.w_ = network_state["W_ei"] * nS
    c_ie.w_ = network_state["W_ie"] * nS
    c_ii.w_ = network_state["W_ii"] * nS

    # Set synaptic delays.
    c_ee.delay = '(1 + 0.5*rand()) * ms'
    c_ei.delay = '(1 + 0.5*rand()) * ms'
    c_ie.delay = '(0.5 + 0.5*rand()) * ms'
    c_ii.delay = '(0.5 + 0.5*rand()) * ms'

    # Create monitors to record neuron activity.
    me_testing = StateMonitor(ge, 'vm', record=True)
    mi_testing = StateMonitor(gi, 'vm', record=True)

    # Run the simulation.
    run(total_time, report=None)
    device.build(directory='standalone_code', compile=True, run=True)

    # Extract and store membrane potential values.
    v_ge = np.array(me_testing.vm / mV)
    v_gi = np.array(mi_testing.vm / mV)
    v = np.vstack((v_ge, v_gi))

    return v, total_time_simulated


def AdEx_test(file_suffix, seed, base_path, directory, test_sequences_csv):
    # Initialize the Brian simulation environment.
    start_scope()

    # List of directories to remove if they exist.
    directories_to_remove = [
        "NetworkBehaviour_test",
        "Activations_test"
    ]
    # Remove existing output directories.
    for dir_name in directories_to_remove:
        dir_path = directory / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path)

    np.random.seed(seed)


    if file_suffix != "features_normalized":
        matrix_path = base_path / f"step_1/X_{file_suffix}.npy"
    else:
        matrix_path = base_path / f"step_0/X_{file_suffix}.npy"

    print(f"Loading feature matrix from: {matrix_path}")
    feature_matrix = np.load(matrix_path)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Load the network state.
    network_state_path = base_path / f"step_2/{file_suffix}.pkl"
    print(f"Loading network state from: {network_state_path}")
    with open(network_state_path, "rb") as f:
        network_state = pickle.load(f)

    # Load test sequences from the CSV file.
    sequences = load_test_sequences_from_csv(test_sequences_csv)

    if not sequences:
        print("No sequences found!")
        return

    # Create output directories.
    network_behaviour_folder = directory / "NetworkBehaviour_test"
    network_activations_folder = directory / "Activations_test"

    subfolder_network = network_behaviour_folder / file_suffix
    subfolder_network.mkdir(parents=True, exist_ok=True)
    subfolder_activations = network_activations_folder / file_suffix
    subfolder_activations.mkdir(parents=True, exist_ok=True)

    # Test each sequence.
    for i, seq_info in enumerate(sequences):

        print(f"\n[{i+1}/{len(sequences)}] Processing sequence...")

        # Test the sequence.
        v, total_time = test_single_sequence(seq_info, feature_matrix, network_state)
        print(f" v shape: {v.shape}")

        # Save the activity matrix with the sequence name.
        seq_name = seq_info['name']
        filename = subfolder_network / f"{seq_name}.npy"
        np.save(filename, v)

        # Remove the initial noise period.
        pre_stim_noise_duration = 100 * ms
        dt = defaultclock.dt
        noise_samples = int(pre_stim_noise_duration / dt)
        v_clean = v[:, noise_samples:]

        # Create a binary activation matrix for the entire clean period.
        binary_matrix = (v_clean > -50.4).astype(int)  # v_t in mV
        print(f" Binary matrix shape: {binary_matrix.shape}")

        activation_data = {
            'binary_matrix': binary_matrix,
            'sequence_name': seq_name,
            'indices': seq_info['indices'],  # Sequence indices
            'shape': binary_matrix.shape
        }

        filename = subfolder_activations / f"{seq_name}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(activation_data, f)

    return total_time


# Load all matrices with metadata from .pkl files in a folder.
def load_matrices_with_metadata(data_dir):

    if not data_dir.exists():
        print(f"ERROR: Folder {data_dir} not found!")
        return {}

    # Find all .pkl files.
    pkl_files = list(data_dir.glob("*.pkl"))

    if not pkl_files:
        print(f"ERROR: No .pkl files found in {data_dir}")
        return {}

    # Load all matrices with metadata into a dictionary.
    matrices_dict = {}

    for file_path in pkl_files:
        with open(file_path, 'rb') as f:
            activation_data = pickle.load(f)

        # Use the sequence name as the dictionary key.
        seq_name = activation_data['sequence_name']
        matrices_dict[seq_name] = activation_data

    return matrices_dict



# Process matrices while preserving the 1:1 correspondence between fragments and indices.
def process_matrices(rep, matrices_dict, n_rows, n_presentations, output_dir):

    # Collect all fragments and their corresponding indices.
    # For each position (1-5), store fragments for each condition (P/C).
    all_fragments = {
        'P': {1: [], 2: [], 3: [], 4: [], 5: []},  # Fragments by position
        'C': {1: [], 2: [], 3: [], 4: [], 5: []},
        'P_indices': {1: [], 2: [], 3: [], 4: [], 5: []},  # Corresponding indices
        'C_indices': {1: [], 2: [], 3: [], 4: [], 5: []}
    }

    for seq_name, activation_data in matrices_dict.items():
        binary_matrix = activation_data['binary_matrix']
        indices = activation_data['indices']
        pattern_str = seq_name.split('_')[-1]

        if len(pattern_str) != rep - 1:
            continue


        # Split the matrix and store each fragment with its index.
        for pos in range(rep):  # pos 0-4 corresponds to positions 1-5
            start_col = pos * n_presentations
            end_col = start_col + n_presentations
            fragment = binary_matrix[:, start_col:end_col]
            idx = indices[pos]
            position_num = pos + 1

            # The first position is assigned to both conditions.
            if pos == 0:
                all_fragments['P'][position_num].append(fragment)
                all_fragments['P_indices'][position_num].append(idx)
                all_fragments['C'][position_num].append(fragment)
                all_fragments['C_indices'][position_num].append(idx)
            else:
                # Positions 2-5 determine the condition.
                pattern_idx = pos - 1
                actual_condition = 'P' if pattern_str[pattern_idx] == 'P' else 'C'

                all_fragments[actual_condition][position_num].append(fragment)
                all_fragments[f'{actual_condition}_indices'][position_num].append(idx)


    # Find the minimum number of fragments per position for each condition.
    min_counts = {}
    for cond in ['P', 'C']:
        counts = [len(all_fragments[cond][pos]) for pos in range(1, rep + 1)]
        min_counts[cond] = min(counts) if counts else 0

    # Build the final matrices.
    results = {}

    for cond in ['P', 'C']:
        if min_counts[cond] == 0:
            print(f"\nNo fragments for condition {cond}")
            continue

        # Final matrix dimensions.
        n_blocks = min_counts[cond]  # Number of virtual sequences

        total_fragments_pos1 = len(all_fragments[cond][1])
        if total_fragments_pos1 >= n_blocks * 2:
            # Use the first n_blocks for P and the second n_blocks for C.
            if cond == 'P': start_idx = 0
            else: start_idx = n_blocks
        else: start_idx = 0

        total_rows = n_blocks * n_rows
        total_cols = rep * n_presentations  # 5 * 3000 = 15000

        # Initialize matrices.
        matrix = np.zeros((total_rows, total_cols))
        indices_matrix = np.zeros((n_blocks, rep), dtype=int)  

        # Iterate over each virtual-sequence block.
        for block_idx in range(n_blocks):
            row_start = block_idx * n_rows
            row_end = (block_idx + 1) * n_rows

            # Iterate over each position (1-5).
            for pos in range(1, rep + 1):
                col_start = (pos - 1) * n_presentations
                col_end = pos * n_presentations

                if pos == 1:
                    # Compute the correct index in the fragment pool.
                    fragment_idx = start_idx + block_idx

                    # Check that the fragment exists.
                    if fragment_idx < len(all_fragments[cond][pos]):
                        fragment = all_fragments[cond][pos][fragment_idx]
                        idx = all_fragments[f'{cond}_indices'][pos][fragment_idx]
                    else:
                        # Fallback: use the default index.
                        fragment = all_fragments[cond][pos][block_idx]
                        idx = all_fragments[f'{cond}_indices'][pos][block_idx]
                else:
                    # Use the default behavior for positions 2-5.
                    fragment = all_fragments[cond][pos][block_idx]
                    idx = all_fragments[f'{cond}_indices'][pos][block_idx]

                # Insert the fragment into the final matrix.
                matrix[row_start:row_end, col_start:col_end] = fragment

                # Store the corresponding index.
                indices_matrix[block_idx, pos - 1] = idx

        # Save the output files.
        np.save(output_dir / f"{cond}_extended.npy", matrix)
        np.save(output_dir / f"{cond}_extended_indices.npy", indices_matrix)

        results[f'{cond}_extended'] = matrix
        results[f'{cond}_indices'] = indices_matrix

    return results
