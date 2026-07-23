import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import pingouin as pg
import pickle
import gc

WILCOXON_COLUMNS = ['neuron_id', 'stimulus_id', 'n_trials', 'mean_baseline',
                    'mean_response', 'p_value', 'significant']


# Build neuron-wise activity dictionaries from activation .pkl files.
def build_neuron_structure_from_pkl(activations_dir, n_neurons, n_timepoints, baseline_window):

    # Initialize one dictionary for primed transitions and one for control transitions.
    priming_by_neuron = {i: {'stim_indices': set(), 'activity': defaultdict(list)}
                         for i in range(n_neurons)}
    control_by_neuron = {i: {'stim_indices': set(), 'activity': defaultdict(list)}
                         for i in range(n_neurons)}

    pkl_files = list(Path(activations_dir).glob("*.pkl"))
    print(f"Found {len(pkl_files)} .pkl files")

    # Use tqdm when available, otherwise fall back to a standard iterator.
    try:
        from tqdm import tqdm
        iterator = tqdm(pkl_files, desc="Processing PKL files")
    except ImportError:
        iterator = pkl_files

    for file_idx, file_path in enumerate(iterator):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        binary_matrix = data['binary_matrix']  # (n_neurons, 15000)
        indices = data['indices']              # [i0, i1, i2, i3, i4]
        seq_name = data['sequence_name']

        # Extract the priming/control pattern from the sequence name suffix.
        pattern = seq_name.split('_')[-1]

        # Iterate over positions 1-4, where each current response has a previous baseline window.
        for pos in range(1, 5):
            current_idx = indices[pos]
            condition = pattern[pos-1]  # 'P' for primed or 'C' for control.

            # Use only the 100 ms immediately before the current response window as baseline. With dt = 0.1 ms this corresponds to 1000 matrix columns.
            start_current = pos * n_timepoints
            end_current = (pos + 1) * n_timepoints
            start_baseline = start_current - baseline_window
            end_baseline = start_current

            # Extract activity for all neurons at once to keep the operation vectorized.
            baseline = binary_matrix[:, start_baseline:end_baseline]
            response = binary_matrix[:, start_current:end_current]

            # Concatenate baseline and response into one transition window per neuron.
            transitions = np.concatenate([baseline, response], axis=1)

            # Route the transition to the dictionary matching the current condition.
            target_dict = priming_by_neuron if condition == 'P' else control_by_neuron

            # Store the transition for each neuron under the corresponding stimulus index.
            for neuron_id in range(n_neurons):
                target_dict[neuron_id]['stim_indices'].add(current_idx)
                target_dict[neuron_id]['activity'][current_idx].append(transitions[neuron_id])

        # Periodically release unused objects during large batch processing.
        if file_idx % 50 == 0:
            gc.collect()

    # Convert stimulus index sets to sorted lists for deterministic downstream processing.
    try:
        from tqdm import tqdm
        neuron_iterator = tqdm(range(n_neurons), desc="Converting sets")
    except ImportError:
        neuron_iterator = range(n_neurons)

    for neuron_id in neuron_iterator:
        priming_by_neuron[neuron_id]['stim_indices'] = sorted(priming_by_neuron[neuron_id]['stim_indices'])
        control_by_neuron[neuron_id]['stim_indices'] = sorted(control_by_neuron[neuron_id]['stim_indices'])

    # Remove neurons without recorded stimulus activity.
    priming_by_neuron = {k: v for k, v in priming_by_neuron.items() if v['stim_indices']}
    control_by_neuron = {k: v for k, v in control_by_neuron.items() if v['stim_indices']}

    return priming_by_neuron, control_by_neuron


# Calculate stimulus-level z-scores for a single neuron using its global baseline activity.
def calculate_z_scores_for_neuron(neuron_dict, baseline_window, response_window):

    if not neuron_dict['stim_indices']:
        return {}

    # Collect baseline samples across all available stimulus transitions.
    all_baselines = []
    for stim_id in neuron_dict['stim_indices']:
        for transition in neuron_dict['activity'][stim_id]:
            all_baselines.extend(transition[:baseline_window])

    if not all_baselines:
        return {}

    all_baselines = np.array(all_baselines)
    mean_baseline_global = np.mean(all_baselines)
    std_baseline_global = np.std(all_baselines)
    if std_baseline_global == 0:
        std_baseline_global = 1.0

    # Compute one z-score per stimulus using the response window mean.
    z_scores = {}
    for stim_id in neuron_dict['stim_indices']:
        stim_responses = []
        for transition in neuron_dict['activity'][stim_id]:
            stim_responses.extend(transition[baseline_window:baseline_window + response_window])

        if stim_responses:
            mean_response = np.mean(stim_responses)
            z_scores[stim_id] = (mean_response - mean_baseline_global) / std_baseline_global

    return z_scores


# Identify neurons with a significant baseline-response change for at least one stimulus, pooling paired trials from the primed and control conditions.
def identify_responsive_neurons(priming_by_neuron, control_by_neuron, baseline_window, response_window, alpha, min_trials_per_stim):

    all_neurons = sorted(set(priming_by_neuron.keys()) | set(control_by_neuron.keys()))
    responsive_neurons = set()
    all_responses = []

    for neuron_id in all_neurons:
        stim_indices = set()
        if neuron_id in priming_by_neuron:
            stim_indices.update(
                priming_by_neuron[neuron_id]['stim_indices']
            )
        if neuron_id in control_by_neuron:
            stim_indices.update(
                control_by_neuron[neuron_id]['stim_indices']
            )

        neuron_has_responsive_stimulus = False

        for stim_idx in sorted(stim_indices):
            baseline_rates = []
            response_rates = []

            for condition_dict in (priming_by_neuron, control_by_neuron):
                if neuron_id not in condition_dict:
                    continue

                transitions = condition_dict[neuron_id]['activity'].get(
                    stim_idx,
                    [],
                )
                for transition in transitions:
                    baseline = transition[:baseline_window]
                    response = transition[
                        baseline_window:baseline_window + response_window
                    ]

                    if baseline.size == 0 or response.size == 0:
                        continue

                    baseline_rates.append(float(np.mean(baseline)))
                    response_rates.append(float(np.mean(response)))

            n_trials = len(baseline_rates)
            if n_trials < min_trials_per_stim:
                continue

            try:
                wilcoxon_result = pg.wilcoxon(
                    x=response_rates,
                    y=baseline_rates,
                )
                p_value = float(wilcoxon_result['p-val'].iloc[0])
            except ValueError:
                # Wilcoxon is undefined when all paired differences are zero.
                continue

            is_significant = bool(
                np.isfinite(p_value) and p_value < alpha
            )
            if is_significant:
                neuron_has_responsive_stimulus = True

            all_responses.append(
                {
                    'neuron_id': neuron_id,
                    'stimulus_id': stim_idx,
                    'n_trials': n_trials,
                    'mean_baseline': np.mean(baseline_rates),
                    'mean_response': np.mean(response_rates),
                    'p_value': p_value,
                    'significant': is_significant,
                }
            )

        if neuron_has_responsive_stimulus:
            responsive_neurons.add(neuron_id)

    results_df = pd.DataFrame(all_responses, columns=WILCOXON_COLUMNS)
    
    return responsive_neurons, results_df


# Rank matched stimuli by control z-score and compare their primed and control responses.
def calculate_rank_analysis_stimulus_matched(priming_by_neuron, control_by_neuron, baseline_window, response_window, max_rank):

    all_ranked_z = []

    try:
        from tqdm import tqdm
        neuron_iterator = tqdm(priming_by_neuron.keys(), desc="Rank analysis")
    except ImportError:
        neuron_iterator = priming_by_neuron.keys()

    for neuron_id in neuron_iterator:
        if neuron_id not in control_by_neuron:
            continue

        # Compute primed and control z-scores for the same neuron.
        z_priming = calculate_z_scores_for_neuron(
            priming_by_neuron[neuron_id], baseline_window, response_window)
        z_control = calculate_z_scores_for_neuron(
            control_by_neuron[neuron_id], baseline_window, response_window)

        if not z_priming or not z_control:
            continue

        # Keep only stimuli available in both conditions.
        common_stimuli = list(set(z_priming.keys()) & set(z_control.keys()))
        if len(common_stimuli) < 2:
            continue

        # Rank stimuli by their control response strength.
        stimuli_with_z = [(stim, z_control[stim]) for stim in common_stimuli]
        stimuli_sorted = sorted(stimuli_with_z, key=lambda x: x[1], reverse=True)

        for rank, (stimulus, zc) in enumerate(stimuli_sorted[:max_rank], 1):
            zp = z_priming[stimulus]

            all_ranked_z.append({
                'Neuron_ID': neuron_id,
                'Stimulus_ID': stimulus,
                'Rank': rank,
                'Z_Priming': round(zp, 3),
                'Z_Control': round(zc, 3),
                'Z_Difference': round(zp - zc, 3)
            })

    if not all_ranked_z:
        return None, None

    df_all_ranked = pd.DataFrame(all_ranked_z)

    # Aggregate z-score statistics by rank across all matched neuron-stimulus pairs.
    rank_stats = df_all_ranked.groupby('Rank').agg({
        'Z_Priming': ['mean', 'std', 'sem', 'count'],
        'Z_Control': ['mean', 'std', 'sem', 'count'],
        'Z_Difference': ['mean', 'std', 'sem', 'count']
    }).round(3)

    rank_stats.columns = ['_'.join(col).strip() for col in rank_stats.columns.values]
    rank_stats = rank_stats.reset_index()

    return df_all_ranked, rank_stats


# Select Wilcoxon-responsive neurons with at least two common stimuli.
def select_neurons_for_rank_analysis(priming_by_neuron, control_by_neuron, baseline_window, response_window):

    all_neurons = sorted(set(priming_by_neuron.keys()) & set(control_by_neuron.keys()))
    valid_neurons = []
    summary_rows = []

    for neuron_id in all_neurons:
        z_priming = calculate_z_scores_for_neuron(
            priming_by_neuron[neuron_id],
            baseline_window,
            response_window,
        )
        z_control = calculate_z_scores_for_neuron(
            control_by_neuron[neuron_id],
            baseline_window,
            response_window,
        )

        common_stimuli = sorted(set(z_priming.keys()) & set(z_control.keys()))
        included = len(common_stimuli) >= 2

        if included:
            valid_neurons.append(neuron_id)

        summary_rows.append(
            {
                'neuron_id': neuron_id,
                'n_priming_stimuli': len(z_priming),
                'n_control_stimuli': len(z_control),
                'n_common_stimuli': len(common_stimuli),
                'included_in_rank_analysis': included,
                'selection_method': 'wilcoxon_then_common_stimuli',
            }
        )

    return set(valid_neurons), pd.DataFrame(summary_rows)


# Run the rank-based Z-score analysis used by optimization loops.
def main_analysis_optimization(activations_dir, save_dir, n_neurons, n_timepoints, baseline_window, response_window, max_rank, alpha=0.05, min_trials_per_stim=10):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    priming_by_neuron, control_by_neuron = build_neuron_structure_from_pkl(
            activations_dir,
            n_neurons,
            n_timepoints,
            baseline_window)
    

    responsive_neurons, wilcoxon_results = identify_responsive_neurons(
            priming_by_neuron,
            control_by_neuron,
            baseline_window,
            response_window,
            alpha,
            min_trials_per_stim)
    
    wilcoxon_results.to_csv(save_dir / 'wilcoxon_test_results.csv', index=False)

    # Exclude non-responsive neurons immediately after the Wilcoxon screen.
    # From this point onward, z-scores and rank statistics are computed only for neurons classified as responsive.
    priming_responsive = {
        n: priming_by_neuron[n]
        for n in responsive_neurons
        if n in priming_by_neuron
    }
    control_responsive = {
        n: control_by_neuron[n]
        for n in responsive_neurons
        if n in control_by_neuron
    }

    valid_neurons, selection_summary = select_neurons_for_rank_analysis(
             priming_responsive,
             control_responsive,
             baseline_window,
             response_window)
    
    priming_valid = {
        n: priming_responsive[n] for n in valid_neurons
    }
    control_valid = {
        n: control_responsive[n] for n in valid_neurons
    }

    df_matched, stats_matched = calculate_rank_analysis_stimulus_matched(
            priming_valid,
            control_valid,
            baseline_window,
            response_window,
            max_rank)
    
    if df_matched is not None:
        df_matched.to_csv(save_dir / 'rank_analysis_data_responsive.csv', index=False)
        df_matched.to_excel(save_dir / 'rank_analysis_data_responsive.xlsx', index=False)

    if stats_matched is not None:
        stats_matched.to_excel(save_dir / 'rank_analysis_stats_responsive.xlsx', index=False)

    return {
        'responsive_neurons': valid_neurons,
        'selection_summary': selection_summary,
        'rank_stats': stats_matched,
    }
